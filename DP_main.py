# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python DP_main.py --model_name swin_v2_b --batch_size 512 --workers 16 --epochs 10 --lr 1e-4
# tensorboard --logdir=/home2/multicam/2024_Multicam/checkpoints/run0621_0339 --port 8800 --bind_all
from datetime import datetime
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default=None, help='model name(efficientnet_v2_s, mobilenet_v3_small, vit_b_16, swin_v2_b)')
parser.add_argument('--weight', type=str, default=None, help='pretrained weight path')
parser.add_argument('--save_snapshot_path', type=str, default=f'./checkpoints/run{datetime.now().strftime("%m%d_%H%M")}', help='save snapshot path')
parser.add_argument('--load_snapshot_path', type=str, default=None)

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# from dataset import face_dataset
from multicam_dataset import MultiCamDataset
from model_zoo import get_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter
# tensorboard_log = './runs/run0830_swin'
tensorboard_log = f'{args.save_snapshot_path}/tensorboard'
# tensorboard --logdir=./runs/run0830_swin --port 8800 --bind_all
# tensorboard --logdir=./runs/run1010_swin --port 8800 --bind_all
# tensorboard --logdir=/home2/multicam/2024_Multicam/checkpoints/run0621_0339/tensorboard --port 8800 --bind_all
import torchvision
# import torchvision.datasets.mnist as mnist


import random
import numpy as np
from tqdm import tqdm

from torchvision import transforms

import time


def decay_learning_rate(init_lr, it, iter_per_epoch, start_epoch, warmup):
    warmup_threshold = warmup
    step = start_epoch * iter_per_epoch + it + 1
    decayed_lr = init_lr * warmup_threshold ** 0.5 * min(step * warmup_threshold**-1.5, step**-0.5)
    return decayed_lr

def save_snapshot(model, optimizer, epoch, path, accuracy):
    snapshot = {
        'epoch': epoch,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'snapshot_{args.model_name}_{epoch}_{accuracy}.pth')
    torch.save(snapshot, path)
    print(f"\tSnapshot saved at {path}")

if __name__ == '__main__':
    seed_number = 999
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model = swin_binary(pretrained=False) if args.model == 'swin' else effi_binary(pretrained=False)
    model, preprocess = get_model(model_name=args.model_name, num_classes=2)
    
    model = nn.DataParallel(model).cuda()
    if args.weight is not None:
        checkpoint = torch.load(args.weight)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        print("load checkpoint: ", args.weight)
    
    # train_dataset = face_dataset('./train', mode='train')
    train_dataset = MultiCamDataset(annotation_path = '/home2/multicam/2024_Multicam/train.json', transform=preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64, pin_memory=True, drop_last=True)
    #32 -> 192
    #64 -> die
    #50 -> 300
    # bs -> 6*bs
    # val_dataset = face_dataset('./validation', mode='val')
    val_dataset = MultiCamDataset(annotation_path = '/home2/multicam/2024_Multicam/val.json', transform=preprocess)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64, pin_memory=True)

    if train_dataloader and val_dataloader:
        print(f'Data load success, train: {len(train_dataloader)}, val: {len(val_dataloader)}')
    
    epochs = 10
    lr = 5e-4
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # scheduler = NoamLR(optimizer, model_size=1024, warmup_steps=100)
    criterion = CrossEntropyLoss()
    

    log_train_loss = []
    log_train_acc = []

    log_val_loss = []
    log_val_acc = []

    summary = SummaryWriter(tensorboard_log)

    for epoch in range(epochs):
        save_epoch = epoch+1
        print('==============Train==============')
        model.train()

        for step, (image, label) in enumerate(tqdm(train_dataloader, leave=True)):
            optimizer.zero_grad()
            
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True)
            output= model(image)
            loss = criterion(output, label)
            # loss = loss.sum()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output, 1)
            acc = torch.sum(pred == label) / len(label)
            
            log_train_loss.append(loss.item())
            log_train_acc.append(acc.item())
            
            

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = decay_learning_rate(lr, save_epoch, len(train_dataloader), save_epoch-1, 8000)


            if step % 500 == 0 and step != 0:
                print('''
                      Train
                      epoch: {}, iter: {}, loss: {:.4f}, acc: {:.4f}
                      '''.format(save_epoch, step, log_train_loss[-1], log_train_acc[-1]))
            
            total_steps = ((save_epoch-1) * len(train_dataloader)) + step + 1
            summary.add_scalar('train_loss_steps', loss, total_steps)
            summary.add_scalar('train_acc_steps', acc, total_steps)     

                
        
        print('''
              Train
              epoch: {}, loss: {:.4f}, acc: {:.4f}
              '''.format(save_epoch, np.mean(log_train_loss), np.mean(log_train_acc)))
        
        print('==============Validation==============')
        model.eval()
        with torch.no_grad():
            for step, (image, label) in enumerate(tqdm(val_dataloader, leave=True)) :
                image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True)
                output= model(image)
                loss = criterion(output, label)
                # loss = loss.sum()
                scheduler.step(loss)
                _, pred = torch.max(output, 1)
                acc = torch.sum(pred == label) / len(label)

                log_val_loss.append(loss.item())
                log_val_acc.append(acc.item())
                if step % 500 == 0 and step != 0:
                    print('''
                        VAlIDATION
                        epoch: {}, iter: {}, loss: {:.4f}, acc: {:.4f}
                        '''.format(save_epoch, step, log_val_loss[-1], log_val_acc[-1]))
        
        print('''
              VAlIDATION
              epoch: {}, loss: {:.4f}, acc: {:.4f}
              '''.format(save_epoch, np.mean(log_val_loss), np.mean(log_val_acc)))
        
        summary.add_scalar('train_loss', np.mean(log_train_loss), save_epoch)
        summary.add_scalar('train_cer', np.mean(log_train_acc), save_epoch)
        summary.add_scalar('valid_loss', np.mean(log_val_loss), save_epoch)
        summary.add_scalar('valid_cer', np.mean(log_val_acc), save_epoch)

        save_snapshot(model, optimizer, save_epoch, args.save_snapshot_path, np.mean(log_val_acc))        
        # checkpoint = {
        #     'epoch': save_epoch,
        #     'model_state_dict': model.module.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     # 'scheduler_state_dict': scheduler.state_dict(),
        # }
        # torch.save(checkpoint, './checkpoints/{}_{}_{}.pth'.format(args.model,save_epoch, np.mean(log_val_acc)))
        # torch.save(checkpoint, './checkpoints/swin_cifar10_{}.pth'.format(save_epoch))

    # print('==============loading test==============')
    # del model
    # time.sleep(5)

    # checkpoint_test = torch.load('./checkpoints/swin_cifar10_{}.pth'.format(save_epoch))
    # start_epoch = checkpoint_test['epoch']
    # print("start_epoch: ",start_epoch)
    # model_test = swin_binary(pretrained=False)
    # model_test = nn.DataParallel(model_test).cuda()
    # model_test.module.load_state_dict(checkpoint_test['model_state_dict'])
    # optimizer_test = Adam(model_test.parameters(), lr=lr)
    # optimizer_test.load_state_dict(checkpoint_test['optimizer_state_dict'])

    # print("optimizer.param_groups[0]['lr']: ", optimizer.param_groups[0]['lr'])
    # print("optimizer_test.param_groups[0]['lr']: ", optimizer_test.param_groups[0]['lr'])


