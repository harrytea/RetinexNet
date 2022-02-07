import os
import argparse
from model import RetinexNet
import wandb
os.environ["WANDB_API_KEY"] = "xxx"
os.environ["WANDB_MODE"] = "offline"
import torch
from datasets import TestDataset
from torch.utils.data import DataLoader
from collections import OrderedDict

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint)
    except:
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu_id', dest='gpu_id', default="4", help='GPU ID (-1 for CPU)')
parser.add_argument('--data_dir', dest='data_dir', default='./data/doc_rec', help='directory storing the test data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir',  default='./ckpts/', help='directory for checkpoints')
parser.add_argument('--ck_epoch', default='99.tar', help='testing epoch')

args = parser.parse_args()



def main(wandb, model):
    '''  dataset  '''
    test_dataset    = TestDataset(args.data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)
    print('Number of testing images: %d' % len(test_dataset))


    '''  load model '''
    decom_path   = os.path.join(args.ckpt_dir, "Decom", args.ck_epoch)
    relight_path = os.path.join(args.ckpt_dir, "Relight", args.ck_epoch)
    load_checkpoint(model.DecomNet, decom_path)
    load_checkpoint(model.RelightNet, relight_path)


    '''  test  '''
    model.eval()
    model.predict(wandb, test_dataloader)


if __name__ == '__main__':

    wandb.init(project="RetinexNet", entity="harrytea", config=args)
    # if not os.path.exists(args.res_dir):
    #     os.makedirs(args.res_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RetinexNet().cuda()
    main(wandb, model)
