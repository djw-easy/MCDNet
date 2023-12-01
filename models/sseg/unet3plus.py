import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import sys, os
import os.path as osp

import torch.nn as nn
import torch.nn.init as init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        weights_init_kaiming(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_init_kaiming(m)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def u3pblock(in_ch, out_ch, num_block=2, kernel_size=3, padding=1, down_sample=False):
    m = []
    if down_sample:
        m.append(nn.MaxPool2d(kernel_size=2))
    for _ in range(num_block):
        m += [nn.Conv2d(in_ch, out_ch, kernel_size, bias=False, padding=padding),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*m)

def en2dec_layer(in_ch, out_ch, scale):
    m = [nn.Identity()] if scale == 1 else [nn.MaxPool2d(scale, scale, ceil_mode=True)]
    m.append(u3pblock(in_ch, out_ch, num_block=1))
    return nn.Sequential(*m)

def dec2dec_layer(in_ch, out_ch, scale, fast_up=True):
    up = [nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else nn.Identity()]
    m = [u3pblock(in_ch, out_ch, num_block=1)]
    if fast_up:
        m = m + up
    else:
        m = up + m  # used in paper
    return nn.Sequential(*m)

        
class FullScaleSkipConnect(nn.Module):
    def __init__(self, 
                 en_channels,   # encoder out channels, high to low
                 en_scales,
                 num_dec,       # number of decoder out
                 skip_ch=64, 
                 dec_scales=None,
                 bottom_dec_ch=1024,
                 dropout=0.3,
                 fast_up=True,):

        super().__init__()
        concat_ch = skip_ch * (len(en_channels) + num_dec)

        # encoder maps to decoder maps connections
        self.en2dec_layers = nn.ModuleList()
        # print(en_scales)
        for ch, scale in zip(en_channels, en_scales):
            self.en2dec_layers.append(en2dec_layer(ch, skip_ch, scale))
        
        # decoder maps to decoder maps connections
        self.dec2dec_layers = nn.ModuleList()
        if dec_scales is None:
            dec_scales = []
            for ii in reversed(range(num_dec)):
                dec_scales.append(2 ** (ii + 1))
        for ii, scale in enumerate(dec_scales):
            dec_ch = bottom_dec_ch if ii == 0 else concat_ch
            self.dec2dec_layers.append(dec2dec_layer(dec_ch, skip_ch, scale, fast_up=fast_up))

        self.droupout = nn.Dropout(dropout)
        self.fuse_layer = u3pblock(concat_ch, concat_ch, 1)

    def forward(self, en_maps, dec_maps=None):
        out = []
        for en_map, layer in zip(en_maps, self.en2dec_layers):
            out.append(layer(en_map))
        if dec_maps is not None and len(dec_maps) > 0:
            for dec_map, layer in zip(dec_maps, self.dec2dec_layers):
                out.append(layer(dec_map))
        return self.fuse_layer(self.droupout(torch.cat(out, 1)))


class U3PEncoderDefault(nn.Module):
    def __init__(self, channels = [3, 64, 128, 256, 512, 1024], num_block=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.downsample_list = nn.Module()
        for ii, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            self.layers.append(u3pblock(ch_in, ch_out, num_block, down_sample= ii > 0))
        self.channels = channels
        self.apply(weight_init)
        
    def forward(self, x):
        encoder_out = []
        for layer in self.layers:
            x = layer(x)
            encoder_out.append(x)
        return encoder_out


class U3PDecoder(nn.Module):
    def __init__(self, en_channels = [64, 128, 256, 512, 1024], skip_ch=64, dropout=0.3, fast_up=True):
        super().__init__()
        self.decoders = nn.ModuleDict()
        en_channels = en_channels[::-1]
        num_en_ch = len(en_channels)
        for ii in range(num_en_ch):
            if ii == 0:
                # first decoding output is identity mapping of last encoder map
                self.decoders['decoder1'] = nn.Identity()
                continue

            self.decoders[f'decoder{ii+1}'] = FullScaleSkipConnect(
                                                en_channels[ii:], 
                                                en_scales=2 ** np.arange(0, num_en_ch-ii),
                                                num_dec=ii, 
                                                skip_ch=skip_ch, 
                                                bottom_dec_ch=en_channels[0],
                                                dropout=dropout,
                                                fast_up=fast_up
                                            )

    def forward(self, enc_map_list:List[torch.Tensor]):
        dec_map_list = []
        enc_map_list = enc_map_list[::-1]
        for ii, layer_key in enumerate(self.decoders):
            layer = self.decoders[layer_key]
            if ii == 0:
                dec_map_list.append(layer(enc_map_list[0]))
                continue
            dec_map_list.append(layer(enc_map_list[ii: ], dec_map_list))
        return dec_map_list


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, 
                 skip_ch=64,
                 dropout=0.3,
                 transpose_final=False,
                 fast_up=True):
        super().__init__()
        channels = [in_channels, 64, 128, 256, 512, 1024]
        self.encoder = U3PEncoderDefault(channels)
        channels = self.encoder.channels
        num_decoders = len(channels) - 1
        decoder_ch = skip_ch * num_decoders

        self.decoder = U3PDecoder(self.encoder.channels[1:], skip_ch=skip_ch, dropout=dropout, fast_up=fast_up)
        self.decoder.apply(weight_init)
        
        if transpose_final:
            self.head = nn.Sequential(
                nn.ConvTranspose2d(decoder_ch, out_channels, kernel_size=4, stride = 2, padding=1, bias=False),
            )
        else:
            self.head = nn.Conv2d(decoder_ch, out_channels, 3, padding=1)
        self.head.apply(weight_init)

    def forward(self, x): 
        _, _, h, w = x.shape
        features = self.encoder(x)
        features = self.decoder(features)

        pred = self.head(features[-1])
        
        return pred


if __name__ == '__main__':
    input = torch.randn((2, 3, 320, 320))
    model = UNet3Plus(num_classes=7)
    out = model(input)