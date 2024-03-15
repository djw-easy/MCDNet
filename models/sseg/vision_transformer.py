import torch
import torch.nn as nn
from torchsummary import summary
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUnet(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=256, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = out_channels
        self.img_size = img_size
        self.zero_head = zero_head

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                        patch_size=4,
                        in_chans=in_channels,
                        num_classes=self.num_classes,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=8,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.2,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUnet(in_channels=3, out_channels=3).to(device)

    x = torch.randn(4, 3, 256, 256).to(device)
    r = model(x)
    print(r.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # summary(model.swin_unet, input_size=(3, 256, 256))
