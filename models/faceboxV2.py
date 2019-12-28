#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from multibox_layer import MultiBoxLayer
import time
from models.acnet_builder import ACBlock, ACBlock_relu
from acnet_fusion1 import convert_acnet_weights1


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, input):
        return torch.cat((F.relu(input), F.relu(-input)), 1)

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception1(nn.Module):
    def __init__(self, deploy=False):
        super(Inception1, self).__init__()

        self.branch0 = BasicConv2d(128, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(128, 32, kernel_size=1, stride=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(128, 24, kernel_size=1, stride=1),
            # BasicConv2d(24, 32, kernel_size=3, stride=1, padding=1)
            ACBlock_relu(24, 32, kernel_size=3, stride=1, padding=1, deploy=deploy)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(128, 24, kernel_size=1, stride=1),
            # BasicConv2d(24, 32, kernel_size=3, stride=1, padding=1),
            ACBlock_relu(24, 32, kernel_size=3, stride=1, padding=1, deploy=deploy),
            # BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
            ACBlock_relu(32, 32, kernel_size=3, stride=1, padding=1, deploy=deploy)
        )

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        output = torch.cat((x0, x1, x2, x3), 1)

        return output


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class Inception(nn.Module):
    def __init__(self):
        super(Inception,self).__init__()
        self.conv1 = nn.Conv2d(128,32,kernel_size=1)
        self.conv2 = nn.Conv2d(128,32,kernel_size=1)
        self.conv3 = nn.Conv2d(128,24,kernel_size=1)
        self.conv4 = nn.Conv2d(24,32,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(128,24,kernel_size=1)
        self.conv6 = nn.Conv2d(24,32,kernel_size=3,padding=1)
        self.conv7 = nn.Conv2d(32,32,kernel_size=3,padding=1)
    def forward(self,x):
        x1 = self.conv1(x)

        x2 = F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        x2 = self.conv2(x2)

        x3 = self.conv3(x)
        x3 = self.conv4(x3)

        x4 = self.conv5(x)
        x4 = self.conv6(x4)
        x4 = self.conv7(x4)

        output = torch.cat([x1,x2,x3,x4],1)
        return output

class FaceBox(nn.Module):
    input_size = 1024
    def __init__(self, deploy=False):
        super(FaceBox, self).__init__()

        #model
        self.conv1 = ACBlock(3,24,kernel_size=7,stride=4,padding=3, deploy=deploy)
        self.conv2 = ACBlock(48,64,kernel_size=5,stride=2,padding=2, deploy=deploy)

        # self.conv1_1 = nn.Conv2d(3,24,kernel_size=3,stride=2,padding=1)
        # self.conv1_1 = BasicConv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.conv1_1 = ACBlock_relu(3, 24, kernel_size=3, stride=2, padding=1, deploy=deploy)
        # self.conv1_2 = nn.Conv2d(24,24,kernel_size=3,stride=2,padding=1)
        # self.conv1_2 = BasicConv2d(24, 24, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = ACBlock_relu(24, 24, kernel_size=3, stride=2, padding=1, deploy=deploy)
        # self.conv1_3 = nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1)
        # self.conv1_3 = BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = ACBlock_relu(24, 24, kernel_size=3, stride=1, padding=1, deploy=deploy)
        # self.bconv1 = nn.Conv2d(72,48,kernel_size=1)
        self.bconv1 = BasicConv2d(72, 48, kernel_size=1)

        self.bn1_1 = nn.BatchNorm2d(24,eps=0.001,momentum=0.1,affine=True)
        self.bn1_2 = nn.BatchNorm2d(24,eps=0.001,momentum=0.1,affine=True)
        self.bn2_1 = nn.BatchNorm2d(64,eps=0.001,momentum=0.1,affine=True)
        self.bn2_2 = nn.BatchNorm2d(64,eps=0.001,momentum=0.1,affine=True)

        # self.conv2_1 = nn.Conv2d(48,64,kernel_size=3,stride=2,padding=1)
        # self.conv2_1 = BasicConv2d(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = ACBlock_relu(48, 64, kernel_size=3, stride=2, padding=1, deploy=deploy)
        # self.conv2_2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        # self.conv2_2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = ACBlock_relu(64, 64, kernel_size=3, stride=1, padding=1, deploy=deploy)
        # self.bconv2 = nn.Conv2d(192,128,kernel_size=1)
        self.bconv2 = BasicConv2d(192, 128, kernel_size=1)

        self.inception1 = Inception1(deploy=deploy)
        self.inception2 = Inception1(deploy=deploy)
        self.inception3 = Inception1(deploy=deploy)

        self.bconv3 = nn.Conv2d(256,128,kernel_size=1)
        self.bconv4 = nn.Conv2d(384,128,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256,eps=0.001,momentum=0.1,affine=True)
        self.bn4 = nn.BatchNorm2d(384,eps=0.001,momentum=0.1,affine=True)

        # self.conv3_1 = nn.Conv2d(128,128,kernel_size=1)
        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1)
        # self.conv3_2 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        # self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = ACBlock_relu(128, 256, kernel_size=3, stride=2, padding=1, deploy=deploy)
        #         # self.conv4_1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1)
        # self.conv4_2 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        # self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = ACBlock_relu(128, 256, kernel_size=3, stride=2, padding=1, deploy=deploy)

        self.multilbox = MultiBoxLayer()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                # nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self,x):
        hs = []
        #x = x.cuda()
        # 原来的
        x1 = self.conv1(x) # 24
        # x1 = self.bn1_1(x1)  # 自己加的
        x1 = torch.cat([x1, -x1],1)
        x1 = F.relu(x1)  # 48
        # 新加的
        x2 = self.conv1_1(x) # 24
        x2 = self.conv1_2(x2)
        x2 = self.conv1_3(x2)
        # x2 = self.bn1_2(x2)
        # x2 = F.relu(x2)  # 24

        x = torch.cat([x1, x2],1) # 72
        x = self.bconv1(x) # 48

        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)

        x3 = self.conv2(x) # 64
        # x3 = self.bn2_1(x3)#自己加的
        x3 = torch.cat([x3, -x3],1)
        x3 = F.relu(x3) # 128

        x4 = self.conv2_1(x) # 64
        x4 = self.conv2_2(x4) # 64
        # x4 = self.bn2_2(x4) # 64
        # x4 = F.relu(x4) # 64

        x = torch.cat([x3, x4],1) # 192
        x = self.bconv2(x) # 128

        x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)# 128通道

        ########     1     ########
        x5 = self.inception1(x)
        x6 = torch.cat([x, x5], 1)
        # x6 = self.bn3(x6)
        x6 = F.relu(x6)
        x6 = self.bconv3(x6)
        x7 = self.inception2(x6)
        x8 = torch.cat([x, x5, x7], 1)
        # x8 = self.bn4(x8)
        x8 = F.relu(x8)
        x8 = self.bconv4(x8)
        x = self.inception3(x8) #128

        # print('x1', x.size())
        hs.append(x)
        x = self.conv3_1(x)  #128
        x = self.conv3_2(x)  #256
        # print('x2', x.size())
        hs.append(x)
        x = self.conv4_1(x)  #128
        x = self.conv4_2(x)  #256
        # print('x3', x.size())
        hs.append(x)
        loc_preds, conf_preds = self.multilbox(hs)
        #loc_preds,loc_landmarks_preds, conf_preds = self.multilbox(hs)

        #return loc_preds,loc_landmarks_preds, conf_preds
        return loc_preds,conf_preds

def timeEval():
    model = FaceBox(deploy=False)
    # model = FaceBox_ori()
    model = model.cuda()
    # model.load_state_dict(torch.load('./weight/faceboxes_209.pt', map_location=lambda storage, loc: storage))
    # model.eval()
    scale = 1024
    batches = [1, 2, 4, 6, 8, 10, 12]
    for batch in batches:
        now = time.time()
        # print('batch:', batch)
        tensor = torch.randn(batch, 3, scale, scale)
        tensor = tensor.cuda()
        data = Variable(tensor, volatile=True)
        lastTime = now - time.time()
        for i in range(50):
            loc, conf = model(data)
        start = time.time()
        for i in range(100):
            loc, conf = model(data)
        end = time.time()
        lastTime = end - start
        print(round((lastTime / 100) * 1000, 2))

def fusionEval():
    model = FaceBox(deploy=True)
    # model = FaceBox()
    # model = FaceBox_ori()
    model = model.cuda()
    model.load_state_dict(torch.load('./faceboxV2_fused.pt', map_location=lambda storage, loc: storage))
    model.eval()
    scale = 1024
    tensor = torch.ones(1, 3, scale, scale)
    tensor = tensor.cuda()
    data = Variable(tensor, volatile=True)
    loc, conf = model(data)
    print(loc, conf)
    # torch.save(model.state_dict(), "faceboxV2_test.pt")

def fusionEval1():
    model = FaceBox(deploy=False)
    # model = FaceBox()
    # model = FaceBox_ori()
    model = model.cuda()
    # model.load_state_dict(torch.load('./faceboxV2_test.pt', map_location=lambda storage, loc: storage))
    # model.eval()
    scale = 1024
    tensor = torch.ones(1, 3, scale, scale)
    tensor = tensor.cuda()
    data = Variable(tensor, volatile=True)
    loc, conf = model(data)
    print(loc, conf)
    # torch.save(model.state_dict(), "faceboxV2_test.pt")

    model1_dict = convert_acnet_weights1(model.state_dict(), "faceboxV2_fused.pt", eps=0)

    model.load_state_dict(model.state_dict())
    model.eval()
    loc, conf = model(data)
    print(loc, conf)
    model1 = FaceBox(deploy=True)
    model1 = model1.cuda()
    model1.load_state_dict(model1_dict)
    model1.eval()
    loc, conf = model1(data)
    print(loc, conf)
    loc, conf = model(data)
    print(loc, conf)

def test():
    model = FaceBox(deploy=False)
    # model = FaceBox()
    # model = FaceBox_ori()
    model = nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('./weight/faceboxV2_wider_2w_1p5_3p2_1p2_91.pt', map_location=lambda storage, loc: storage))
    model.eval()
    scale = 1024
    tensor = torch.ones(1, 3, scale, scale)
    tensor = tensor.cuda()
    loc, conf = model(tensor)
    print(loc, conf)

    model1_dict = convert_acnet_weights1(model.state_dict(), "faceboxV2_fused.pt", eps=0)

    model1 = FaceBox(deploy=True)
    model1 = nn.DataParallel(model1)
    model1 = model1.cuda()
    model1.load_state_dict(model1_dict)
    model1.eval()
    loc, conf = model1(tensor)
    print(loc, conf)

    # model1 = FaceBox(deploy=True)
    # model1 = nn.DataParallel(model1)
    # model1 = model1.cuda()
    # model1.load_state_dict(torch.load('./weight/faceboxV2_wider_2w_1p5_3p2_1p2_91_fused.pt', map_location=lambda storage, loc: storage))
    # model1.eval()
    # loc, conf = model1(tensor)
    # print(loc, conf)

if __name__ == '__main__':
    fusionEval1()
