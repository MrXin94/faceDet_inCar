#encoding:utf-8
'''
txt描述文件 image_name.jpg num x y w h 1 x y w h 1 这样就是说一张图片中有两个人脸
'''
import os
import sys
import os.path

import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
#import imgaug as ia
#from imgaug import augmenters as iaa
from math import *
import glob
import lmdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2

from encoderl import DataEncoder

class ListDataset(data.Dataset):
    image_size=1024

    def __init__(self,root, data_type,list_file,lmdb_path,train,transform, multi_scale=False):
        print('data init')
        self.root = root
        self.data_type = data_type
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = [] # 这里的box是左上角和右下角！！！ x1 y1 x2 y2
        self.landmarks = []
        self.lmdb_path = lmdb_path
        self.labels = []
        self.small_threshold = 20./self.image_size  # face that small than threshold will be ignored
        self.data_encoder = DataEncoder()
        self.snames = glob.glob('./picture/shuiyin/*')
        self.num_shuiyin = len(self.snames)
        self.multi_scale = multi_scale

        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            num_faces = int(splited[1])
            box=[]
            label=[]
            landmarks = []
            if num_faces <= 0:
                continue
            self.fnames.append(splited[0])
            for i in range(num_faces):
                #x = float(splited[2+5*i])
                #y = float(splited[3+5*i])
                #w = float(splited[4+5*i])
                #h = float(splited[5+5*i])
                #c = int(splited[6+5*i])
                x = float(splited[2+15*i])
                y = float(splited[2+1+15*i])
                w = float(splited[2+2+15*i])
                h = float(splited[2+3+15*i])
                landmarks_ = []
                for j in range(5):
                    pt_x = float(splited[2+3+15*i+j*2+1])
                    pt_y = float(splited[2+3+15*i+j*2+2])
                    landmarks_.append(pt_x)
                    landmarks_.append(pt_y)
                c = float(splited[2+15*i+14])
                box.append([x,y,x+w,y+h])
                landmarks.append(landmarks_)
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
            self.landmarks.append(torch.Tensor(landmarks))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        while True:
            if self.multi_scale:
                if isinstance(idx, int):
                    self.image_size = 1024
                else:
                    self.image_size= idx[1]
                    idx = idx[0]
            fname = self.fnames[idx]
            if self.data_type == "img":
                img = cv2.imread(os.path.join(self.root,fname))
                # print(fname)
            elif self.data_type == "lmdb":
                img = self.getImg(fname)
                # print(fname)
            #print img.shape
            if img is not None:
                # print self.root,fname
                assert img is not None

                boxes = self.boxes[idx].clone()
                labels = self.labels[idx].clone()
                landmarks = self.landmarks[idx].clone()

                if self.train:
                    img = self.pad_to_square(img)
		    assert img is not None
                    # img, boxes, landmarks = self.random_rot(img, boxes,landmarks)
                    if random.random() > 0.7:
                        img, boxes, landmarks, labels = self.random_crop_edge(img, boxes,landmarks,labels)
                    else:
                        #img, boxes, labels = self.random_crop(img, boxes, labels)
                        img, boxes,landmarks,labels = self.random_crop(img, boxes,landmarks,labels)
                    #img = self.random_bright(img)
                    # img = self.my_random_bright(img)
                    #img, boxes = self.random_flip(img, boxes)
                    #if img is None:
                    #    print fname
                    img, boxes ,landmarks = self.random_flip(img, boxes,landmarks)
                    #img = self.addImage(img)
                    boxwh = boxes[:,2:] - boxes[:,:2]
                    #print('boxwh', boxwh)

                h,w,_ = img.shape
                self.data_encoder = DataEncoder(float(self.image_size))
                #print img
                img = cv2.resize(img,(self.image_size,self.image_size))

                #坐标归一化操作
                for i in range(boxes.shape[0]):
                    if boxes[i][0] == boxes[i][2]:
                        boxes[i][0] += 2
                    if boxes[i][1] == boxes[i][3]:
                        boxes[i][1] += 2
                landmarks  /= torch.Tensor([w,h]*5).expand_as(landmarks)
                boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
                for t in self.transform:
                    img = t(img)
                #print boxes
                try:
                        #loc_target,conf_target = self.data_encoder.encode(boxes,labels)
                        loc_target,landmarks_target, conf_target = self.data_encoder.encode(boxes,landmarks, labels, fname)
                        #print loc_target
                        break
                except:
                        idx = random.randint(0,self.num_samples)
                        continue
        #print img.shape
        return img,loc_target,landmarks_target,conf_target
    
    def getImg(self, ckey):
        lmdb_path = self.lmdb_path
        env = lmdb.open(lmdb_path, max_dbs=8, map_size=int(1e12), readonly=True, lock = False)
        txn = env.begin()
        cbuf = txn.get(ckey)
        arr = np.fromstring(cbuf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    
    def pad_to_square(self,image):
            if random.random() > 0.5:
                return image
            #print 'pad to square'
            factor = random.random()+1
            height, width, _ = image.shape
            height_new = int(height*factor)
            width_new = int(width*factor)
            image_new = np.zeros((height_new,width_new,3),dtype = 'uint8')
            image_new[0:0 + height, 0:0 + width] = image
            return image_new

    def random_getim(self):
        idx = random.randrange(0,self.num_samples)
        fname = self.fnames[idx]
        img = self.getImg(fname)
        #img = cv2.imread(os.path.join(self.root,fname))
        boxes = self.boxes[idx].clone()
        landmarks = self.landmarks[idx].clone()
        labels = self.labels[idx]

        return img, boxes, landmarks,labels

    def __len__(self):
        return self.num_samples

    # 一个点参照中心进行旋转
    def point_rot(self, xy, org_center, angle):
        org = xy - org_center
        a = np.deg2rad(angle)
        new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a),
                        -org[0] * np.sin(a) + org[1] * np.cos(a)])
        return new

    def random_rot(self, image, boxes,landmarks):
        if np.random.rand()>0.5:
            #angle = np.float(20 * np.random.rand() - 10)
            angle = np.float(40 * np.random.rand() - 20)
            #angle = np.float(90 * np.random.rand() - 45)
            im_rot = self.cv2_rot(image, angle)
            h, w, _ = im_rot.shape
            org_center = (np.array(image.shape[:2][::-1]) - 1) / 2.
            rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.
            for i in range(boxes.shape[0]):
                box = np.array(boxes[i])
                landmarks_ = np.array(landmarks[i])
                temp = []
                temp.extend(self.point_rot(box[0:2], org_center, angle) + rot_center)
                temp.extend(self.point_rot(np.array([box[2], box[1]]), org_center, angle) + rot_center)
                temp.extend(self.point_rot(np.array([box[0], box[3]]), org_center, angle) + rot_center)
                temp.extend(self.point_rot(box[2:4], org_center, angle) + rot_center)
                x1 = max(min(temp[::2]),0)# 左上角坐标
                y1 = max(min(temp[1::2]),0)# 不超过边界
                x2 = min(max(temp[::2]), w)# 右下角坐标
                y2 = min(max(temp[1::2]), h)
                boxes[i,:] = torch.Tensor(np.array([x1, y1, x2, y2]))

                #rotate landmarks
                for j in range(5):
                    pt_xy = self.point_rot(landmarks_[j*2:(j*2+2)],org_center,angle) + rot_center
                    pt_x = max(min(pt_xy[0],w),0)
                    pt_y = max(min(pt_xy[1],h),0)
                    landmarks_[j*2] = pt_x
                    landmarks_[j*2+1] = pt_y
                landmarks[i,:] = torch.Tensor(landmarks_)
            #print 'rot: ',angle
            return im_rot, boxes,landmarks
        else:
            return image, boxes,landmarks
    #用cv2旋转图片
    def cv2_rot(self, img, degree):
        height, width = img.shape[:2]
        # 旋转后的尺寸
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
        matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
        return imgRotation

    # 对图片随机添加水印
    def addImage(self, img):
        if np.random.rand()>0.5:
            idx = np.random.randint(0, self.num_shuiyin)
            sname = self.snames[idx]
            img_s = cv2.imread(sname)
            # 把旧的img_s当成mask，然后转换成多颜色的
            if np.random.rand()>0.5:
                img_color = cv2.imread('./picture/color.jpg')
                h, w, _ = img_s.shape
                img_color = cv2.resize(img_color, (w, h))
                img2gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 25, 255, cv2.THRESH_BINARY)
                img_s = cv2.bitwise_and(img_color, img_color, mask=mask)
            h, w, _ = img.shape
            # 函数要求两张图必须是同一个size
            hs, ws, _ = img_s.shape
            if h/hs < w/ws:
                shape_h = np.random.randint(int(h / 4), h)
                shape_w = int(shape_h*ws/hs)
            else:
                shape_w = np.random.randint(int(w / 4), w)
                shape_h = int(shape_w*hs/ws)
            img2 = cv2.resize(img_s, (shape_w, shape_h))
            # alpha，beta，gamma可调
            beta = 0.2*np.random.rand()
            gamma = 0
            loc_h = np.random.randint(0, h - shape_h)
            loc_w = np.random.randint(0, w - shape_w)
            img[loc_h:loc_h + shape_h, loc_w:loc_w + shape_w, :] = cv2.addWeighted(
                img[loc_h:loc_h + shape_h, loc_w:loc_w + shape_w, :], 1, img2, beta, gamma)
        return img

    def random_flip(self, im, boxes,landmarks):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin_boxes = w - boxes[:,2]
            xmax_boxes = w - boxes[:,0]
            boxes[:,0] = xmin_boxes
            boxes[:,2] = xmax_boxes

            landmarks_ = landmarks.clone()
            for j in range(5):
                landmarks_[:,j*2] = w - landmarks_[:,j*2]

            #new pt0
            landmarks[:,0] = landmarks_[:,2]
            landmarks[:,1] = landmarks_[:,3]
            #new pt1
            landmarks[:,2] = landmarks_[:,0]
            landmarks[:,3] = landmarks_[:,1]
            #new pt2 
            landmarks[:,4] = landmarks_[:,4]
            landmarks[:,5] = landmarks_[:,5]
            #new pt3
            landmarks[:,6] = landmarks_[:,8]
            landmarks[:,7] = landmarks_[:,9]
            #new pt4
            landmarks[:,8] = landmarks_[:,6]
            landmarks[:,9] = landmarks_[:,7]
            #print 'ranom flip'
            return im_lr, boxes,landmarks

        return im, boxes,landmarks


    def random_crop_edge(self,im, boxes,landmarks, labels):

        while True:
            # choose bbox
            max_area = -1
            best_bbox = None
            for i in range(boxes.size()[0]):
                bbox = boxes[i,:]
                x1,y1,x2,y2 = bbox
                w = x2-x1+1
                h = y2-y1+1
                area = w*h
                if area > max_area:
                    max_area = area
                    best_bbox = bbox
            #
            w = best_bbox[2] - best_bbox[0]
            h = best_bbox[3] - best_bbox[1]
            ratio = float(h) / w
            if ratio > 2:
                im, boxes, landmarks, labels = self.random_getim()
                #print '循环'
                continue
            elif w <= 0 or h <= 0:
                im, boxes, landmarks, labels = self.random_getim()
                #print '循环1'
                continue
            elif float(w)/im.shape[1] < self.small_threshold or float(h)/im.shape[0] < self.small_threshold:
                im, boxes, landmarks, labels = self.random_getim()
                continue
            else:
                break
        boxes_ori = boxes.clone()
        labels_ori = labels.clone()
        landmarks_ori = landmarks.clone()
        im_ori = im.copy()

        imh, imw, _ = im.shape
        center_x = int((best_bbox[0]+best_bbox[2])/2)
        center_x = max(center_x,0)
        center_x = min(imw-1,center_x)
        if best_bbox[0] > imw/2:
            x2 = center_x
            if x2  < imh:
                x1 = 0
                y1 = random.randint(0,imh-x2)
                x2 = x2
                y2 = y1 + x2
            else:
                y1 = 0
                x1 = x2 - imh+1
                x2 = x2
                y2 = imh-1
            #
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(x2,imw-1)
            y2 = min(y2,imh-1)

            im_new =im[y1:y2,x1:x2,:]
            boxes[:,0] = boxes[:,0] -x1
            boxes[:,1] = boxes[:,1] -y1
            boxes[:,2] = boxes[:,2] -x1
            boxes[:,3] = boxes[:,3] -y1
        else:
            x1 = center_x
            if (imw-x1)  < imh:
                x1 = x1
                y1 = random.randint(0,imh-(imw-x1))
                x2 = imw-1
                y2 = y1 + (x2-x1)
            else:
                y1 = 0
                x1 = x1
                x2 = x1+imh-1
                y2 = imh-1
            #
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(x2,imw-1)
            y2 = min(y2,imh-1)

            im_new =im[y1:y2,x1:x2,:]
            boxes[:,0] = boxes[:,0] -x1
            boxes[:,1] = boxes[:,1] -y1
            boxes[:,2] = boxes[:,2] -x1
            boxes[:,3] = boxes[:,3] -y1

        #delete boxes not in this croped Img
        index = []
        for i in range(boxes.size()[0]):
            x1,y1,x2,y2 = boxes[i]
            #if x1 < 0 and y1 < 0 and x2 >= im.shape[1] and y2 >= im.shape[0]:
            if (x1 <= 0 and x2 <= 0) or (x1>=im_new.shape[1] and x2 >= im_new.shape[1] ):
                index.append(0)
            elif (y1 <= 0 and y2 <= 0) or (y1>=im_new.shape[0] and y2 >= im_new.shape[0] ):
                index.append(0)
            elif ((x2-x1)/im_new.shape[1] < self.small_threshold) or ( (y2-y1)/im_new.shape[0] < self.small_threshold ):
                index.append(0)
            else:
                index.append(1)

        if sum(index)<=0:
            return im_ori, boxes_ori, landmarks_ori, labels_ori
        
        #print torch.LongTensor(index).nonzero().size(),'*****'
        index = torch.LongTensor(index).nonzero().squeeze(1)
        #print 'index nonzero: ',index.nonzero().squeeze(1)
        boxes = boxes.index_select(0,index)
        landmarks = landmarks.index_select(0,index)
        labels = labels.index_select(0,index)
        h,w,_ = im_new.shape
        boxes[:,0].clamp_(min=0, max=w)
        boxes[:,1].clamp_(min=0, max=h)
        boxes[:,2].clamp_(min=0, max=w)
        boxes[:,3].clamp_(min=0, max=h)

        w = boxes[:,2] - boxes[:,0]
        id = torch.sum(w<=0)
        if id > 0:
            print( boxes[:,2] - boxes[:,0],'w',boxes[:,2])
        h = boxes[:,3] - boxes[:,1]
        id = torch.sum(h<=0)
        if id > 0:
            print( boxes[:,3] - boxes[:,1],'h',boxes[:,3])

        return im_new, boxes, landmarks, labels




    def random_crop(self, im, boxes,landmarks, labels):
        imh, imw, _ = im.shape
        short_size = min(imw, imh)
        while True:
            mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
            #mode = random.choice([None, 0.3])
            if 0:
                    if mode is None:
                        boxes_uniform = boxes / torch.Tensor([imw,imh,imw,imh]).expand_as(boxes)
                        #landmarks_uniform = landmarks / torch.Tensor([imw,imh]*5)
                        boxwh = boxes_uniform[:,2:] - boxes_uniform[:,:2]
                        mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold)
                        if not mask.any():
                            #print('default image have none box bigger than small_threshold')
                            im, boxes, landmarks, labels = self.random_getim()
                            imh, imw, _ = im.shape
                            short_size = min(imw,imh)
                            continue
                        selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                        selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                        selected_landmarks = landmarks.index_select(0,mask.nonzero().squeeze(1))
                        return im, selected_boxes,selected_landmarks, selected_labels

            #ratio = random.randint(5,20) / 10.0
            if mode is None:
                w = short_size
                h = w
            else:
                w = random.randrange(int(0.3*short_size), short_size)
                h = w
            for kk in range(10):
                #h = min(int(ratio*w),short_size-1)
                #print imw-w+1,imh-h+1
                x = random.randrange(imw - w+1)
                y = random.randrange(imh - h+1)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2
                roi2 = roi.expand(len(center), 4)
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])
                mask = mask[:,0] & mask[:,1]
                if not mask.any():
                    #im, boxes, landmarks, labels = self.random_getim()
                    #imh, imw, _ = im.shape
                    #short_size = min(imw,imh)
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_landmarks = landmarks.index_select(0,mask.nonzero().squeeze(1))

                img = im[y:y+h,x:x+w,:]
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                for k in range(5):
                    selected_landmarks[:,2*k].add_(-x).clamp_(min=0,max=w)
                    selected_landmarks[:,2*k+1].add_(-y).clamp_(min=0,max=h)
                # print('croped')

                boxes_uniform = selected_boxes / torch.Tensor([w,h,w,h]).expand_as(selected_boxes)
                #landmarks_uniform = selected_landmarks / torch.Tensor([imw,imh]*5)
                boxwh = boxes_uniform[:,2:] - boxes_uniform[:,:2]
                mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold)
                if not mask.any():
                    #print('crop image have none box bigger than small_threshold')
                    #im, boxes, landmarks, labels = self.random_getim()
                    #imh, imw, _ = im.shape
                    #short_size = min(imw,imh)
                    continue
                selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_landmarks_selected = selected_landmarks.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                #print 'random crop '
                return img, selected_boxes_selected,selected_landmarks_selected, selected_labels
            im, boxes, landmarks, labels = self.random_getim()
            imh, imw, _ = im.shape
            short_size = min(imw,imh)    

    def my_random_bright(self, img):
        alpha = np.random.rand()+0.5
        beta = np.random.randint(0,3)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if beta == 0:
            image = ImageEnhance.Brightness(image).enhance(alpha)#亮度
        elif beta == 1:
            image = ImageEnhance.Contrast(image).enhance(alpha)#对比度
        else:
            image = ImageEnhance.Color(image).enhance(alpha)#色度
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return img

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im
    def distort(self,image):
            def _convert(image, alpha=1, beta=0):
                tmp = image.astype(float) * alpha + beta
                tmp[tmp < 0] = 0
                tmp[tmp > 255] = 255
                image[:] = tmp

            if random.random() < 0.5:
                return image
            image = image.copy()

            if random.randrange(2):

                #brightness distortion
                if random.randrange(2):
                    _convert(image, beta=random.uniform(-32, 32))

                #contrast distortion
                if random.randrange(2):
                    _convert(image, alpha=random.uniform(0.5, 1.5))

                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                #saturation distortion
                if random.randrange(2):
                    _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

                #hue distortion
                if random.randrange(2):
                    tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                    tmp %= 180
                    image[:, :, 0] = tmp

                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            else:

                #brightness distortion
                if random.randrange(2):
                    _convert(image, beta=random.uniform(-32, 32))

                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                #saturation distortion
                if random.randrange(2):
                    _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

                #hue distortion
                if random.randrange(2):
                    tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                    tmp %= 180
                    image[:, :, 0] = tmp

                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

                #contrast distortion
                if random.randrange(2):
                    _convert(image, alpha=random.uniform(0.5, 1.5))

            return image
    def testGet(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root,fname))
        #cv2.imwrite('test_encoder_source.jpg', img)
        #cv2.imshow('ori img',img)
        boxes = self.boxes[idx].clone()
        landmarks = self.landmarks[idx].clone()
        # print(boxes)
        labels = self.labels[idx].clone()

        #for box in boxes:
        #    cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255))
        #cv2.imwrite(fname, img)

        if self.train:
            img = self.pad_to_square(img)
            img, boxes,landmarks = self.random_rot(img, boxes,landmarks)
            img, boxes, landmarks, labels = self.random_crop_edge(img, boxes,landmarks,labels)
            #img, boxes,landmarks, labels = self.random_crop(img, boxes,landmarks, labels)
            #img = self.random_bright(img)
            #img, boxes,landmarks = self.random_flip(img, boxes,landmarks)
        

        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        landmarks  /= torch.Tensor([w,h]*5).expand_as(landmarks)

        img = cv2.resize(img,(self.image_size,self.image_size))
        for t in self.transform:
            img = t(img)

        #print(idx, fname, boxes)

        return img, boxes,landmarks,labels

if __name__ == '__main__':
    file_root = '/nfs/project'
    file_root = '/var/log'
    #train_dataset = ListDataset(root=file_root,list_file='./widerFace_bbox_five_landmarks.txt',train=True,transform = [transforms.ToTensor()] )
    #train_dataset = ListDataset(root=file_root,list_file='/nfs/project/faceData_InCar/test_label1_new.txt',train=True,transform = [transforms.ToTensor()] )
    train_dataset = ListDataset(root=file_root,list_file='./widerFace_InCar.txt',train=True,transform = [transforms.ToTensor()] )
    print('the dataset has %d image' % (len(train_dataset)))
    for i in range(len(train_dataset)):
        print(i)
        item = random.randrange(0, len(train_dataset))
        item = item
        img, boxes,landmarks, labels = train_dataset.testGet(item)
        # img, boxes = train_dataset[item]
        img = img.numpy().transpose(1,2,0).copy()*255
        #train_dataset.data_encoder.test_encode(boxes, img, labels)

        boxes = boxes.numpy().tolist()
        landmarks = landmarks.numpy().tolist()
        w,h,_ = img.shape
        # print('img', img.shape)
        # print('boxes', boxes.shape)
        img = img.astype('uint8')
        print( img.shape)
        for box in boxes:

             
            x1 = int(box[0]*w)
            y1 = int(box[1]*h)
            x2 = int(box[2]*w)
            y2 = int(box[3]*h)
            print( x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255))
            boxw = x2-x1
            boxh = y2-y1
            #print(boxw,boxh, box)
            if boxw is 0 or boxh is 0:
                raise( 'zero width')
            #continue
        #for landmarks_ in landmarks:
        #    idxs = [0,3,1,2,4]
        #    #for j in range(1):
        #    for j in idxs:
        #        pt_x = int(landmarks_[j*2]*w)
        #        pt_y = int(landmarks_[j*2+1]*h)
        #        cv2.circle(img,(pt_x,pt_y),2,(0,0,255))
        cv2.imshow('img',img)
        cv2.waitKey()
        
