from multicam_dataset import MultiCamDataset, get_files_with_extension
from model_zoo import get_model
# ['efficientnet_v2_s', 'mobilenet_v3_small', 'vit_b_16', 'swin_v2_b', 'resnet18']
from argparse import ArgumentParser
from datetime import datetime
parser = ArgumentParser()
# parser.add_argument('--device', type=str, default='1, 2', help='cuda devices', required=True)
parser.add_argument('--model_name', type=str, default=None, help='model name(efficientnet_v2_s, mobilenet_v3_small, vit_b_16, swin_v2_b)')
parser.add_argument('--weight', type=str, default=None, help='pretrained weight path')
parser.add_argument('--save_snapshot_path', type=str, default=f'./checkpoints/run{datetime.now().strftime("%m%d_%H%M")}', help='save snapshot path')
parser.add_argument('--load_snapshot_path', type=str, default=None)

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--full_train', type=bool, default=True, help='full train')
args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 main.py --model mobilenet_v3_large \
# --batch_size 128 --workers 64 --epochs 50 --lr 1e-4

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 DDP_main.py --model swin_v2_b \
--save_snapshot_path ./checkpoints/run0621_swin_v2_b_full_train2 \
--batch_size 64 --workers 16 --epochs 50 --lr 1e-4 --full_train True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 DDP_main.py --model swin_v2_b \
--save_snapshot_path ./checkpoints/run0621_swin_v2_b_tail_train \
--batch_size 64 --workers 16 --epochs 50 --lr 1e-4 --full_train False
'''

# BATCH MEMORY
# efficientnet_v2_s: 64BS / 32423MiB / 6gpu
# mobilenet_v3_small: 
#                   64BS / 2973MiB / 6gpu 
#                   128BS / 4259MiB / 6gpu 
#                   512BS / 11687MiB / 6gpu 
#                   768BS / 16645MiB / 6gpu
#                   1024BS / 20305MiB / 6gpu
#                   1536BS / 29476MiB / 6gpu     
#                   2048BS / 41655MiB / 6gpu
# mobilnet_v3_large:
#                   32BS / 3485MiB / 6gpu
#                   256BS / 14855MiB / 6gpu
#                   512BS / 27689MiB / 6gpu
#                   BS / MiB / 6gpu
#                   BS / MiB / 6gpu
# vit_b_16
#                   32BS / 7095MiB / 6gpu
#                   64BS / 11231MiB / 6gpu
#                   128BS / 18925MiB / 6gpu
#                   256BS / 34619MiB / 6gpu
# swin_v2_b
#                   32BS / 15011MiB / 6gpu
#                   64BS / 27401MiB / 6gpu
#                   80BS / 33067MiB / 6gpu


import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss

import os
import socket
import random
import numpy as np

import torchvision
from tqdm import tqdm

from tensorboardX import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"Rank {os.environ['RANK']}, Local Rank {os.environ['LOCAL_RANK']}, World Size {os.environ['WORLD_SIZE']}")

# def find_free_port():
#     """Finds a free port on localhost."""
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(('', 0))

#         return s.getsockname()[1]

# def ddp_setup():
#     # os.environ["MASTER_ADDR"] = "127.0.0.1"
#     os.environ["MASTER_PORT"] = str(find_free_port())  # 동적으로 포트 할당
#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#     print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
#     print(f"Rank {os.environ['RANK']}, Local Rank {os.environ['LOCAL_RANK']}, World Size {os.environ['WORLD_SIZE']}")

def load_snapshot(model, optimizer, scheduler, path):
    loc = f'cuda:{os.environ["LOCAL_RANK"]}'  
    snapshot = torch.load(path, map_location=loc)
    epoch = snapshot['epoch']
    model.load_state_dict(snapshot['model'])
    optimizer.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
    print(f"Snapshot loaded from {path}")
    return epoch, model, optimizer, scheduler

def save_snapshot(model, optimizer, scheduler, epoch, path, accuracy):
    snapshot = {
        'epoch': epoch,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'snapshot_{args.model_name}_{epoch}_{accuracy}.pth')
    torch.save(snapshot, path)
    print(f"\tSnapshot saved at {path}")

def main():
    ddp_setup()
    set_seed(999)

    if int(os.environ['LOCAL_RANK']) == 0:
        tensorboard_log = os.path.join(args.save_snapshot_path, 'logs')

    # if int(os.environ['LOCAL_RANK']) == 0:
    #     print(f"Data pathes loading...")
    #     train_data_pathes = get_files_with_extension('/home2/multicam/AIHUB_LIP/Train_Frames/', '.jpg')
    #     val_data_pathes = get_files_with_extension('/home2/multicam/AIHUB_LIP/Validation_Frames', '.jpg')
    
    # torch.distributed.barrier()
    local_rank = int(os.environ['LOCAL_RANK'])

    # train_data_pathes = [train_data_pathes]
    # val_data_pathes = [val_data_pathes]
    # torch.distributed.broadcast_object_list(train_data_pathes, src=0, device=torch.device(f'cuda:{local_rank}'))
    # torch.distributed.broadcast_object_list(val_data_pathes, src=0, device=torch.device(f'cuda:{local_rank}'))

    # train_data_pathes = train_data_pathes[0]
    # val_data_pathes = val_data_pathes[0]

    # print(f"Data pathes loaded: train {len(train_data_pathes)}, val {len(val_data_pathes)}")

    
    start_epoch = 0

    model, preprocess = get_model(args.model_name, num_classes=2, full_train=args.full_train)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
    criterion = CrossEntropyLoss()
    

    model.to(local_rank)

    if args.load_snapshot_path is not None:
        start_epoch, model, optimizer, scheduler = load_snapshot(model, optimizer, scheduler, args.load_snapshot_path)
        
    model = DDP(module = model, device_ids=[local_rank],) 

    # train_dataset = MultiCamDataset(root_dir='/home2/multicam/AIHUB_LIP/Train_Frames/', transform=preprocess)
    train_dataset = MultiCamDataset(annotation_path = '/home2/multicam/2024_Multicam/train.json', transform=preprocess)
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)


    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                                batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # val_dataset = MultiCamDataset(root_dir='/home2/multicam/AIHUB_LIP/Validation_Frames', transform=preprocess)
    val_dataset = MultiCamDataset(annotation_path = '/home2/multicam/2024_Multicam/val.json', transform=preprocess)
    # val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, 
                            # sampler=val_sampler,
                            batch_size=args.batch_size,  num_workers=args.workers, pin_memory=True)
    print(f"Data loaded: train {len(train_dataset)}, val {len(val_dataset)}")

    torch.distributed.barrier()
    print(f"Model Ready")

    model.train()
    # while True:
    for epoch in range(start_epoch+1, args.epochs+1):
        train_sampler.set_epoch(epoch)

        for x, y in tqdm(train_loader):
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, | Loss: {loss.item()}")

        torch.distributed.barrier()
        if epoch % 3 == 0 and local_rank == 0:
            model.eval()
            with torch.no_grad():
                total = 0
                correct = 0
                for x, y in tqdm(val_loader):
                    x, y = x.to(local_rank), y.to(local_rank)
                    y_pred = model(x)
                    val_loss = criterion(y_pred, y)
                    scheduler.step(val_loss)
                    _, predicted = torch.max(y_pred, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            accuracy = correct / total
            model.train()
            print(f"\tEpoch: {epoch}, | Accuracy: {accuracy}")

        if local_rank == 0:
            save_snapshot(model, optimizer, scheduler, epoch, args.save_snapshot_path, accuracy)


if __name__ == "__main__":
    # train_data_pathes = get_files_with_extension('/home2/multicam/AIHUB_LIP/Train_Frames/', '.jpg')
    # val_data_pathes = get_files_with_extension('/home2/multicam/AIHUB_LIP/Validation_Frames', '.jpg')
    main()
