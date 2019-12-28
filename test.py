#coding:utf8
from models.facebox_noBn_rfb import FaceBox
from encoderl import DataEncoder

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import os
import time
import numpy as np
from eval.eval import eval

cuda = True
scale = 640
imgDir = './all_test'
# imgDir = '../insightface/RetinaFace/hard_sample_test_voc'
label_path = './6000_label.txt'
# label_path = './2000_label.txt'
outfile = './faceboxV1.1_noBn-6k-640-predict.txt'
# label_file = './eval/6000_label.txt'
# predict_file = './eval/predict/faceboxV1.1-6k-1024-predict.txt'
output_recall_far_file = './recall/faceboxV1.1_noBn-6k-640-recall1.txt'

def drawImg(img,boxes,probs):
    #h,w,_ = img.shape
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    #max_size = max(h,w)
    face_num = len(boxes)/4
    s = str(int(face_num)) + " "
    for i in range(int(face_num)):
        x1 = int(boxes[i*4]*1)
        y1 = int(boxes[i*4+1]*1)
        x2 = int(boxes[i*4+2]*1)
        y2 = int(boxes[i*4+3]*1)
        # s = s + str(x1) + " " +str(y1) + " " +str(x2) + " " +str(y2) + " "
    # print(s[:-1])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
        # cv2.putText(img, str(probs[i]), (x1,y1), font, 0.4, (0,255,0))
    # cv2.imshow('img',img)
    # cv2.waitKey()

    return img

def test_img(net,data_encoder,imgPath,scale=640,thresh=0.7):
    assert os.path.exists(imgPath)
    img = cv2.imread(imgPath)
    img_ori = img.copy()
    if img is None:
        print("can not open image:")
        return [],[]
    #preprocess: pading img to square with zero values
    h,w,_ = img.shape
    long_size = max(h,w)
    img_ = np.zeros((long_size,long_size,3),dtype='uint8')
    img_[:h,:w,:] = img
    img = img_.copy()
    h,w,_ = img.shape

    #detect face and landmarks
    now = time.time()
    img = cv2.resize(img, (scale,scale))
    resize_time = time.time() - now
    img = img.astype('float32')
    img = img / 255.0
    im_tensor = torch.from_numpy(img.transpose((2,0,1)))
    im_tensor = torch.unsqueeze(im_tensor, 0)
    if cuda:
        im_tensor = im_tensor.cuda()
    loc, conf = net(Variable(im_tensor, volatile=True))

    loc = loc.data.squeeze(0)

    conf = F.softmax(conf.squeeze(0)).data

    boxes_,_, probs_ = data_encoder.decode(loc,conf,thresh)

    boxes = []
    probs = []

    boxes_ = boxes_.numpy()
    probs_ = probs_.numpy()
    num = boxes_.shape[0]
    for j in range(num):
        boxes.append(boxes_[j][0]*long_size)
        boxes.append(boxes_[j][1]*long_size)
        boxes.append(boxes_[j][2]*long_size)
        boxes.append(boxes_[j][3]*long_size)
        probs.append(probs_[j])
  
    #drawImg(img_ori,boxes,probs)
    forward_time = time.time() - now
    return boxes,probs,resize_time, forward_time

def test_faceInCar():
    net = FaceBox()
    # scale = 640
    # scale = 720
    # scale = 800
    # scale = 960

    if cuda:
        net = net.cuda()
    net.load_state_dict(torch.load('./weight/faceboxes_333.pt', map_location=lambda storage, loc:storage))
    net.eval()
    data_encoder = DataEncoder(float(scale))



    # outfile = './1-3_predict.txt'
    # outfile = './faceboxV1.1_noBn-6k-1024-predict.txt'
    # outfile = './faceboxV1.1-6k-800-predict.txt'
    # outfile = './faceboxV1.1-6k-960-predict.txt'
    # outfile = './faceboxV1.1-6k-1024-predict.txt'
    fout = open(outfile,'w')

    lines = open(label_path,'r').readlines()

    resize_time_total = 0
    forward_time_total = 0
    for i,line in enumerate(lines):
        if i % 5 == 0:
            print (i,len(lines))
        imgName = line.strip().split(' ')[0]
        imgName = imgName.split("/")[-1]
        imgPath = os.path.join(imgDir,imgName)
        boxes,probs,resize_time, forward_time = test_img(net,data_encoder,imgPath,scale,thresh=0.1)
        resize_time_total += resize_time
        forward_time_total += forward_time
        fout.write(imgName+' ')

        face_num = len(boxes)/4
        fout.write(str(face_num)+' ')
        for i in range(int(face_num)):
            x1 = int(boxes[i*4]*1)
            y1 = int(boxes[i*4+1]*1)
            x2 = int(boxes[i*4+2]*1)
            y2 = int(boxes[i*4+3]*1)
            prob = str(probs[i])
            fout.write(str(x1)+' '+ str(y1)+' '+ str(x2)+' '+str(y2)+' '+prob+' ')
        fout.write('\n')
    fout.close()

    # print(resize_time_total / len(lines))
    # print(forward_time_total / len(lines))



def test():
    # imgPath = './test2.jpg'
    #imgPath = './badcase-2k.jpg'
    #imgPath = './t.png'
    #imgPath = './noface2.png'
    net = FaceBox()
    scale = 1024
    #scale = 640
    if cuda:
        net = net.cuda()
    net.load_state_dict(torch.load('./weight/faceboxes_209.pt', map_location=lambda storage, loc:storage))
    net.eval()
    data_encoder = DataEncoder(float(scale))

    file_path = "./all_test/"
    files = [file_name for file_name in os.listdir(file_path) if file_name.lower().endswith('jpg')]
    for file in files:
        imgPath = os.path.join(file_path, file)
        img = cv2.imread(imgPath)

        boxes, probs = test_img(net, data_encoder, img, scale, thresh=0.5)
        # print(boxes)
        face_num = len(boxes) / 4
        s = str(int(face_num)) + " "
        for i in range(int(face_num)):
            x1 = int(boxes[i * 4] * 1)
            y1 = int(boxes[i * 4 + 1] * 1)
            x2 = int(boxes[i * 4 + 2] * 1)
            y2 = int(boxes[i * 4 + 3] * 1)
            s = s + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " "
        print(file, s[:-1])
        # cv2.imwrite("./all_test_detected/" + file + "_result.jpg", img)
    #test img
    # if 0:
    #     files = os.listdir('./eval/badcase_edge/')
    #     for file in files:
    #         imgPath  = os.path.join('./eval/badcase_edge/',file)
    #         print(imgPath)
    #         t = time.time()
    #         for k in range(30):
    #             boxes,probs = test_img(net,data_encoder,imgPath,scale,thresh=0.5)
    #         print( 'each time: ',(time.time()-t)/30.0)

if __name__ == '__main__':
    test_faceInCar()

    eval(label_path, outfile, output_recall_far_file)
    #test()


