import os, json
import datetime
import argparse


class Options:
    def __init__(self, model_name: str):
        models = ["unet", "cloudnet", "deeplabv3plus", "cdnetv2", "segnet", "hrnet", "mscff", "mfcnn", "swinunet", "mcdnet"]
        assert model_name in models
        parser = argparse.ArgumentParser('DJW -- Cloud Detection')
        # params of data storage
        parser.add_argument("--root", type=str, default="./data/l8", help="absolute path of the dataset")
        parser.add_argument("--cloudy", type=str, default="cloudy", help="dir name of the cloudy image dataset")
        parser.add_argument("--dc", type=str, default="bccr", help="dir name of the thin cloud removal dataset")
        parser.add_argument("--label", type=str, default="label", help="dir name of the label dataset")
        # params of dataset
        parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
        parser.add_argument("--in_channels", type=int, default=3, help="number of image channels")
        parser.add_argument("--num_classes", type=int, default=3, help="number of classes")
        parser.add_argument("--file_suffix", type=str, default='.TIF', help="the filename suffix of train data")
        # params of model training
        parser.add_argument("--start_epoch", type=int, default=1, help="start epoch")
        parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
        # other
        parser.add_argument("--checkpoint", type=str, default='0', help="checkpoint to load pretrained models")
        parser.add_argument("--model_name", type=str, default=model_name, help="model name")
        parser.add_argument("--save_name", type=str, default=model_name, help="dir name to save model")
        parser.add_argument(
            "--sample_interval", type=int, default=1, help="epoch interval between sampling of images from model"
        )
        parser.add_argument(
                "--evaluation_interval", type=int, default=1, help="epoch interval between evaluation from model"
        )
        parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
        parser.add_argument("--time", type=str, default=self._time(), help="the run time")

        self.args = parser.parse_args([])

        os.makedirs(os.path.join(self.args.root, 'show', self.args.save_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.root, "saved_models", self.args.save_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.root, "args", self.args.save_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.root, "evaluation", self.args.save_name), exist_ok=True)

    def parse(self, save_args=True):
        print(self.args)
        if save_args:
            self._save_args()
        return self.args
    
    def _time(self):
        now = datetime.datetime.now()
        year = str(now.year)
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        hour = str(now.hour).zfill(2)
        date_str = year + month + day + hour
        return date_str
    
    def _save_args(self):
        out_path = os.path.join(self.args.root, "args", 
                                self.args.save_name, 
                                f'{self.args.time}.args')
        with open(out_path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

