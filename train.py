import os
import time
import torch
from torch.utils.data import DataLoader

from utils.config import Options
from models.model_zoo import get_model
from utils.trainers import BaseTrainer
from utils.metric import evaluate, sample_images
from datasets.cloud_dection import ImageDataset, ForeverDataIterator


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


def save_model(model, args, name):
    torch.save(
        {'model': model.state_dict()}, 
        os.path.join(args.root, "saved_models", f"{args.save_name}/{args.time}_{name}.pth")
    )


def main(args):
    # Initialize MODEL
    model = get_model(args, device)

    if args.checkpoint != '0':
        # Load pretrained models
        checkpoint = torch.load(os.path.join(args.root, f"saved_models/{args.save_name}/{args.checkpoint}.pth"))
        model.load_state_dict(checkpoint['model'])

    trainer = BaseTrainer(args, model, device)

    # Configure dataloaders
    train_loader = DataLoader(
        ImageDataset(args, mode="train", normalization=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu, 
        pin_memory=True
    )
    test_loader = DataLoader(
        ImageDataset(args, mode="test", normalization=True),
        batch_size=args.batch_size,
        num_workers=1,
        drop_last=True
    )
    test_loader = ForeverDataIterator(test_loader, device)

    #  Training
    acc = 0
    for epoch in range(args.start_epoch, args.n_epochs+1):
        model.train()
        trainer.train(epoch, train_loader)
        model.eval()

        # If at sample interval save image
        if args.sample_interval and epoch % args.sample_interval == 0:
            sample_images(args, test_loader, model, epoch)

        # If at sample interval evaluation
        if args.evaluation_interval and epoch % args.evaluation_interval == 0:
            acc_ = evaluate(args, model, device, epoch)
            if acc_ > acc:
                acc = acc_
                save_model(model, args, name=f'best_acc')

        if epoch > 1 and args.checkpoint_interval and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            save_model(model, args, name=epoch)


if __name__ == '__main__':
    # time.sleep(2 * 60 * 60)
    args = Options(model_name='mcdnet').parse(save_args=True)
    
    main(args)
    
