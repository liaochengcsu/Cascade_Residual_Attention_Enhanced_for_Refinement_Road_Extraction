import torch
# import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import cv2
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from data_agu import Mydataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from thop import profile
# from nextmodel import AttDinkNet34
# from mod import AttDinkNet34
# from model_best import AttDinkNet34
from net import OurDinkNet50
from seg_iou import mean_IU,frequency_weighted_IU,r_iou,Acc_Metric
from loss import dice_bce_loss_with_logits1, dice_bce_loss_with_logits,binary_cross_logits

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tf.logging.set_verbosity(tf.logging.INFO)
#pretrained_path = r""

class args:

    train_path = r'D:\Data\Road\massachusetts\training\train/train.csv'
    val_path = r'D:\Data\Road\massachusetts\validate\valid/test.csv'
    num_test_img = 399

    result_dir = 'Result/'
    batch_size = 6
    learning_rate = 0.01
    max_epoch = 100
best_train_acc = 0.4
now_time = datetime.now()
time_str = datetime.strftime(now_time,'%m-%d_%H-%M-%S')

log_dir = os.path.join(args.result_dir,time_str)
if not os.path.exists(log_dir):
     os.makedirs(log_dir)

writer = SummaryWriter(log_dir)

normMean = [0.314031, 0.354777, 0.351035]
normStd = [0.179488, 0.179823, 0.188222]

normTransfrom = transforms.Normalize(normMean, normStd)
transform = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom
    ])

train_data = Mydataset(path=args.train_path,transform=transform,augment=True)
val_data = Mydataset(path=args.val_path,transform=transform,augment=False)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

print("train data set:",len(train_loader)*args.batch_size)
print("valid data set:",len(val_loader)*args.batch_size)

net = OurDinkNet50()
net.cuda()


class dice_loss(nn.Module):
    def forward(self, input, target):
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return -torch.log((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
if torch.cuda.is_available():
    w = torch.Tensor([1, 2]).cuda()
    # continue training...
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load(pretrained_path)
    # net = net.to(device)
    # net.load_state_dict(checkpoint['state_dict'])
else:
    w = torch.Tensor([1, 2])

criterion1 = dice_bce_loss_with_logits1().cuda()
criterion2 = dice_bce_loss_with_logits().cuda()

optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, dampening=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True,min_lr=0.00001)
savepath=r"D:\Projects\Road_seg_deep\Result\temp"

#---------------------------4、训练网络---------------------------
for epoch in range(args.max_epoch):
    loss_sigma = 0.0
    loss_val_sigma = 0.0
    acc_val_sigma = 0.0
    net.train()

    for i,data in enumerate(train_loader):
        inputs, labels,lab_name = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        labels = labels.float().cuda()
        optimizer.zero_grad()
        outputs, lower = net.forward(inputs)
        # outputs=torch.sigmoid(outputs)
        outputs=torch.squeeze(outputs,dim=1)
        lower = torch.squeeze(lower, dim=1)
        lossr = criterion2(labels, lower)
        losss = criterion1(labels, outputs)
        loss = losss+4*lossr

        loss.backward()
        optimizer.step()

        loss_sigma += loss.item()
        if i % 50 == 0 and i>0 :
            loss_avg = loss_sigma /50
            loss_sigma = 0.0
            print("Training:Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss:{:.4f}".format(
                epoch + 1, args.max_epoch,i+1,len(train_loader),loss_avg))
            writer.add_scalar("LOSS", loss_avg, epoch)
            # scheduler.step(loss_avg)
            # writer.add_scalar("LEARNING_RATE", scheduler.get_lr()[0], epoch)

    if epoch%1==0:

        tmp_save_name = os.path.join(savepath, str(epoch))
        if not os.path.exists(tmp_save_name):
            os.mkdir(tmp_save_name)
        # tmp_save_namer = os.path.join(savepath, str(epoch)+'r')
        # if not os.path.exists(tmp_save_namer):
        #     os.mkdir(tmp_save_namer)

        net.eval()
        acc_val_recall = 0
        acc_val_precision = 0
        acc_val_f1 = 0
        acc_val_iou = 0

        acc_val = 0
        data_list = []
        for i, data in enumerate(val_loader):
            inputs, labels, img_name = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.float().cuda()
            with torch.no_grad():
                # deep, outputs, lower =
                predicts,_ = net.forward(inputs)

            predicts = torch.sigmoid(predicts)
            # resultr = 255*np.squeeze(predicts)
            predicts[predicts < 0.5] = 0
            predicts[predicts >= 0.5] = 1
            result = np.squeeze(predicts)
            # outputs = torch.squeeze(outputs, dim=1)

            cc = labels.shape[0]
            for index in range(cc):
                # acc_val_sigma += r_iou(labels[index].cpu().detach().numpy(), result[index].cpu().detach().numpy())
                recall, precision, f1, iou = Acc_Metric(result[index].cpu().detach().numpy(),labels[index].cpu().detach().numpy())
                acc_val_recall += recall
                acc_val_precision += precision
                acc_val_f1 += f1
                acc_val_iou += iou
                cv2.imwrite(os.path.join(tmp_save_name, img_name[index]), result[index].cpu().detach().numpy()*255)

        # 保存模型
        val_acc_recall = acc_val_recall / args.num_test_img
        val_acc_precision = acc_val_precision / args.num_test_img
        val_acc_f1 = acc_val_f1 / args.num_test_img
        val_acc_iou = acc_val_iou / args.num_test_img
        print("valid_recall:", val_acc_recall,"valid_precision:", val_acc_precision, "valid_f1:", val_acc_f1, "valid_iou:", val_acc_iou)

        print("lr:",args.learning_rate)
        print("last_best_iou:", best_train_acc)
        scheduler.step(val_acc_iou)
        if (val_acc_iou) > best_train_acc:
            # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            state = {'state_dict': net.state_dict()}
            filename = os.path.join(log_dir, str(epoch) + '_checkpoint-best.pth')
            torch.save(state, filename)

            best_train_acc = val_acc_iou
        else:
            shutil.rmtree(tmp_save_name)

writer.close()
net_save_path = os.path.join(log_dir,'net_params_end.pkl')
torch.save(net.state_dict(),net_save_path)