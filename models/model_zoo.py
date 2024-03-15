import torch
from models.sseg.unet import UNet
from models.sseg.cloudnet import CloudNet
from models.sseg.deeplabv3_plus import DeepLabV3Plus
from models.sseg.cdnetv2 import CDnetV2
from models.sseg.segnet import SegNet
from models.sseg.hrnet import HighResolutionNet
from models.sseg.mscff import MSCFF
from models.sseg.mfcnn import MFCNN
from models.sseg.vision_transformer import SwinUnet
from models.sseg.mcdnet import MCDNet


def get_model(args, device):
    model_name = args.model_name
    in_channels, out_channels = args.in_channels, args.num_classes
    if model_name == "unet":
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "cloudnet":
        model = CloudNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "segnet":
        model = SegNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "cdnetv2":
        model = CDnetV2(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "hrnet":
        model = HighResolutionNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "mscff":
        model = MSCFF(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "mfcnn":
        model = MFCNN(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "swinunet":
        model = SwinUnet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "mcdnet":
        model = MCDNet(in_channels=in_channels, out_channels=out_channels)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    model = model.to(device)
    if model_name == "mcdnet":
        inputs = [torch.randn(args.batch_size, args.in_channels, args.img_size, args.img_size, device=device), 
                  torch.randn(args.batch_size, args.in_channels, args.img_size, args.img_size, device=device)]
    else:
        inputs = [torch.randn(args.batch_size, args.in_channels, args.img_size, args.img_size, device=device)]
    model(*inputs)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("%s Params: %.2fM" % (model_name, params_num / 1e6))

    return model
