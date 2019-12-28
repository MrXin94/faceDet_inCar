# coding:utf8
import os, argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from multibox_loss import MultiBoxLoss
from dataset import ListDataset
from batch_sampler import BatchSampler,RandomSampler


import numpy as np
import time
import datetime

parser = argparse.ArgumentParser(description='Train Facebox')
# general
parser.add_argument('--network', help='network name', default="facebox", type=str)
parser.add_argument('--dataset_type', help='dataset type lmdb or img', default="lmdb", type=str)
parser.add_argument('--lmdb', help='lmdb name', default="../images_mdb", type=str)
parser.add_argument('--img_path', help='images path', default='../data/images', type=str)
parser.add_argument('--label', help='label file', default='widerface_2w_1p5w_3p2w-label.txt', type=str)
parser.add_argument('--scale', help='training scale', default=1024, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--batch', help='batch size', default=64, type=int)
parser.add_argument('--start_epoch', help='start epoch', default=1, type=int)
parser.add_argument('--end_epoch', help='end epoch', default=600, type=int)
parser.add_argument('--fineturn', help='fineturning or not', default=False, type=bool)
parser.add_argument('--model_path', help='fineturning model weight',
                    default="weight/faceboxV1_wider_2w_1p5_3p2_1p2_321.pt", type=str)
parser.add_argument('--multi_scale', help='multi_scale training or not', default=False, type=bool)
parser.add_argument('--conf_weight', help='conf_loss weight', default=1, type=int)

args = parser.parse_args()

use_gpu = torch.cuda.is_available()
lmdb_path = args.lmdb
file_root = args.img_path
if args.dataset_type in ["lmdb", "img"]:
    data_type = args.dataset_type
else:
    print("==unknown data_type==")
label_path = args.label
network = args.network
scale = args.scale
learning_rate = args.lr
end_epoch = args.end_epoch
batch_size = args.batch
start_epoch = args.start_epoch
multi_scale = args.multi_scale

print(args)
if network == "facebox":
    print(network)
    from models.facebox import FaceBox
elif network == "facebox_noBn":
    from models.facebox_noBn import FaceBox
elif network == "facebox_rfb":
    from models.facebox_noBn_rfb import FaceBox
elif network == "facebox_noBn_rfb":
    from models.facebox_noBn_rfb import FaceBox
elif network == "faceboxV2":
    from models.faceboxV2 import FaceBox
elif network == "faceboxV2_noBn":
    from models.faceboxV2_noBn import FaceBox
else:
    print("==unknown model==")

net = FaceBox()
# net = FaceBox_ori()
net = nn.DataParallel(net)


if args.fineturn:
    # model_path = os.path.join('./weight', 'facebox'+network+'_wider_2w_1p5_' + str(start_epoch) + '.pt')
    model_path = args.model_path
    pretrained_dict = torch.load(model_path)
    model_dict = net.state_dict()
    model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
    model_dict.update(pretrained_dict)
    net.load_state_dict(pretrained_dict)

if use_gpu:
    net.cuda()

# print('load model...')
# net.load_state_dict(torch.load('weight/faceboxes.pt'))

criterion = MultiBoxLoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
# optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)

train_dataset = ListDataset(root=file_root, data_type=data_type, list_file=label_path,
                            lmdb_path=lmdb_path, train=True, transform=[transforms.ToTensor()], multi_scale=multi_scale)
if not multi_scale:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12)
else:
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(RandomSampler(train_dataset),
                                                       batch_size,
                                                       True,
                                                       multiscale_step=1,
                                                       img_sizes=list(range(640, 1024 + 1, 128))), pin_memory=True, num_workers=12)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))

num_iter = 0
# vis = visdom.Visdom()
# win = vis.line(Y=np.array([0]), X=np.array([0]))
# win_lr = vis.line(Y=np.array([learning_rate]),X=np.array([0]))
net.train()

loss_file = './total_loss.txt'
f_write = open(loss_file, 'w')
f_write.write('step     loss     learning_rate')
f_write.write('\n')
f_write.close()

# 每30个周期学习率*0.9
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
# 在指定周期学习率下降
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,200], gamma=0.1)
# 每10个周期loss不下降，学习率降低
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

for epoch in range(start_epoch, end_epoch):
    f_write = open(loss_file, 'a+')

    print('\n\nStarting epoch %d / %d' % (epoch + 1, end_epoch))
    print('Learning Rate for this epoch: {}'.format(optimizer.param_groups[0]['lr']))

    total_loss = 0.
    total_conf_loss = 0.
    total_loc_loss = 0.
    total_landmarks_loss = 0.
    start_time = time.time()
    for i, (images, loc_targets, landmarks_targets, conf_targets) in enumerate(train_loader):
        images = Variable(images)
        loc_targets = Variable(loc_targets)
        if landmarks_targets is not None:
            landmarks_targets = Variable(landmarks_targets)
        conf_targets = Variable(conf_targets)
        if use_gpu:
            images, loc_targets, conf_targets = images.cuda(), loc_targets.cuda(), conf_targets.cuda()
        # loc_preds,loc_landmarks_preds, conf_preds = net(images)
        loc_preds, conf_preds = net(images)

        # loc_loss ,loc_landmarks_loss, conf_loss = criterion(loc_preds,loc_targets,loc_landmarks_preds,landmarks_targets,conf_preds,conf_targets)
        loc_loss, _, conf_loss = criterion(loc_preds, loc_targets, None, None, conf_preds, conf_targets)
        # loss = loc_loss + loc_landmarks_loss*5 + conf_loss*1
        loss = loc_loss + conf_loss * args.conf_weight
        total_loss += loss.data
        total_conf_loss += conf_loss.data
        total_loc_loss += loc_loss.data
        # total_landmarks_loss += loc_landmarks_loss.data[0]
        # total_landmarks_loss += loc_landmarks_loss.data[0]*100
        total_landmarks_loss = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num_iter + 1) % 20 == 0:
            print(
                '%s Epoch [%d/%d], Iter [%d/%d] conf Loss: %.4f, average_conf_loss: %.4f, loc Loss: %.4f, average_loc_loss: %.4f  average_time: %.4f step/s,'
                % (datetime.datetime.now(), epoch + 1, end_epoch, i + 1, len(train_loader), conf_loss.data,
                   total_conf_loss / (i + 1), loc_loss.data, total_loc_loss / (i + 1),
                   50 / (time.time() - start_time)))
            # %(datetime.datetime.now(),epoch+1, end_epoch, i+1, len(train_loader), conf_loss.data[0], total_conf_loss / (i+1),loc_loss.data[0], total_loc_loss / (i+1),loc_landmarks_loss.data[0]*100, total_landmarks_loss / (i+1), 50/(time.time()-start_time)))

            start_time = time.time()
            num_iter = num_iter + 1
            f_write.write('%d     %.4f    %f' % (num_iter, total_loss / (i + 1), optimizer.param_groups[0]['lr']))
            f_write.write('\n')
        if 0:
            vis.line(Y=np.array([total_loss / (i + 1)]), X=np.array([num_iter]),
                     win=win,
                     update='append')
            vis.line(Y=np.array([optimizer.param_groups[0]['lr']]), X=np.array([num_iter]),
                     win=win_lr,
                     update='append')
        else:
            num_iter = num_iter + 1
        # break
    print('%s Epoch [%d/%d], average_conf_loss: %.4f,  average_loc_loss: %.4f  '
          % (datetime.datetime.now(), epoch + 1, end_epoch, total_conf_loss / len(train_loader),
             total_loc_loss / len(train_loader)))

    if not os.path.exists('weight/'):
        os.mkdir('weight')
    print('saving model ...')
    model_name = 'weight/facebox'+network+'_'+label_path+'_' + str(epoch + 1) + '.pt'
    # model_name = 'weight/faceboxes_ori_'+ str(epoch+1) +'.pt'
    torch.save(net.state_dict(), model_name)
    print(model_name)

    f_write.close()



