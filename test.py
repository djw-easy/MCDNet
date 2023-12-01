import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.model_zoo import get_model
from torch.utils.data import DataLoader
from datasets.cloud_dection import ImageDataset

from utils.trainers import data_prefetcher
from utils.metric import get_pred, Evaluator


def get_args(model_name, checkpoint, dataset='l8', num_classes=3):
    parser = argparse.ArgumentParser('DJW -- Cloud Detection')
    # params of data storage
    parser.add_argument("--root", type=str, default=f"./data/{dataset}", help="absolute path of the dataset")
    parser.add_argument("--cloudy", type=str, default="cloudy", help="dir name of the cloudy image dataset")
    parser.add_argument("--dc", type=str, default="dccr", help="dir name of the thin cloud removal dataset")
    parser.add_argument("--label", type=str, default="label", help="dir name of the label dataset")
    # params of dataset
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--in_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--num_classes", type=int, default=num_classes, help="number of classes")
    parser.add_argument("--file_suffix", type=str, default='.TIF', help="the filename suffix of train data")
    # other
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--checkpoint", type=str, default=checkpoint, help="checkpoint to load pretrained models")
    parser.add_argument("--model_name", type=str, default=model_name, help="model name")
    parser.add_argument("--save_name", type=str, default=model_name, help="dir name to save model")
    args = parser.parse_args([])
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    args = get_args('mcdnet', '0', dataset='l8', num_classes=3)

    model = get_model(args=args, device=device)
    checkpoint = torch.load(os.path.join(args.root, f"saved_models/{args.save_name}/{args.checkpoint}.pth"))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    dataset = ImageDataset(args, mode="test", normalization=True)
    test_loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True
    )

    evaluator = Evaluator(args.num_classes, device, average='micro')
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        prefetcher = data_prefetcher(test_loader)
        cloudy, dc, label = prefetcher.next()
        while cloudy is not None:
            predict = get_pred(args, model, cloudy, dc)
            predict = torch.argmax(predict, dim=1)
            predict = predict.to(device, dtype=torch.long)

            evaluator.send(predict, label)
            cloudy, dc, label = prefetcher.next()
            pbar.update(1)
        pbar.close()

    metrics = evaluator.result()
    df = pd.DataFrame(data=metrics[np.newaxis, :], columns=evaluator.indicators)
    print(df)




