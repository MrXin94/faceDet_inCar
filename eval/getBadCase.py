# coding:utf8


import os
import numpy as np
import pickle
import cv2


def judge(bbox0,bbox1,thresh=0.5):
    match = False
    maxX = max(bbox0[0],bbox1[0])
    maxY = max(bbox0[1],bbox1[1])
    minX = min(bbox0[2],bbox1[2])
    minY = min(bbox0[3],bbox1[3])
    IOU = (minX-maxX+1)*(minY-maxY+1)

    w0 = bbox0[2] - bbox0[0] + 1
    h0 = bbox0[3] - bbox0[1] + 1

    w1 = bbox1[2] - bbox1[0] + 1
    h1 = bbox1[3] - bbox1[1] + 1

    IOU /= float(( w0*h0 + w1*h1 - IOU +0.000001))
    if IOU > thresh:
        match = True
    return match


def analysis(data,outfile):
    ''' 计算不同阈值下，模型的召回和误检 '''

    imgNames = np.array(data['imgNames'])
    predict_bboxs = np.array(data['predict_bboxs'])
    scores = np.array(data['scores'])
    match_or_nots = np.array(data['match_or_nots'])
    face_nums = np.array(data['face_nums'])
    total_face_nums = data['total_face_nums']

    fout = open(outfile,'w')
    print(imgNames)
    print(match_or_nots)
    for i in range(imgNames.shape[0]):
        if scores[i] >= 0.5 and not match_or_nots[i]:
            print("./hard_sample_test/"+imgNames[i])
            img = cv2.imread("./hard_sample_test/"+str(imgNames[i]))
            cv2.imread("badcase-2k/"+imgNames[i], img)
            # cv2.rectangle(img, (predict_bboxs[i][0], predict_bboxs[i][1]), (predict_bboxs[i][2], predict_bboxs[i][3]), (255,0,0), 2)
            # fout.write("{} {}\n".format(imgNames[i], predict_bboxs[i]))

    #计算不同阈值的召回、误伤数目，以及漏召回、误伤的case
    # for thresh in range(0,100,1):
    #     thresh  /= 100.0
    #
    #     total_predict_num = np.sum(scores>=thresh)
    #     total_recall_num = np.sum(  (scores>=thresh) & (match_or_nots) )
    #     total_false_accept_num = total_predict_num - total_recall_num
    #     print( thresh,total_predict_num,total_recall_num,total_false_accept_num,total_face_nums)
    #     fout.write(str(thresh)+' '+ str(total_recall_num)+ ' '+str(total_false_accept_num)+' '+str(total_face_nums)+'\n')
    fout.close()

    return

def checkBadCase(thresh = 0.5):
    ''' check哪些图片有漏召回，误检测 '''
    #data_file = './eval_data.pkl'
    #fin = open(data_file,'rb')
    #data = pickle.load(fin)
    ##data = {'imgNames':imgNames,'predict_bboxs':predict_bboxs,'scores':scores,'match_or_nots':match_or_nots,'face_nums':face_nums,'total_face_nums':total_face_nums}

    label_file = '/nfs/project/faceData_InCar/test_label2_new.txt'
    #predict_file = '/Users/didi//workspace/project/FaceDetection/data/face_inCar/labelImgs/after_human_check/test/test_0_facebox.txt'
    predict_file = './predicts_209.txt'
    #predict_file = './predicts_209_1024.txt'
    reverseImg_file = './reverseImg.txt'

    #imgDir = '/Users/didi//workspace/project/FaceDetection/data/face_inCar/labelImgs/after_human_check/test/test_0'
    imgDir = '/nfs/project'
    thresh = thresh
    iou_thresh = 0.5
    imgOutDir = './badcase-2k/'

    lines0 = open(label_file,'r').readlines()
    lines1 = open(predict_file,'r').readlines()

    assert  len(lines0) == len(lines1)

    reverseImgs = open(reverseImg_file,'r').readlines()
    reverseImgs = [line.strip() for line in reverseImgs]

    num = len(lines0)
    cnt = 0
    for i in range(num):
        #if i < 4000:
        #    continue
        items0 = lines0[i].strip().split(' ')
        items1 = lines1[i].strip().split(' ')
        face_num_label = int(items0[1])
        face_num_predict_ = int(items1[1])
        face_num_predict = 0
        for k in range(face_num_predict_):
            score = float(items1[2 + k * 5 + 4])
            if score > thresh:
                face_num_predict += 1

        #if face_num_label == face_num_predict:
        #if face_num_label >= face_num_predict:
        if face_num_label <= face_num_predict:
            continue

        name = items0[0].split('/')[-1]
        name = name.rsplit('_',1)[0]
        if name in reverseImgs:
            continue
 
        #show predict and labels

        imgPath = os.path.join(imgDir,items0[0])
        img = cv2.imread(imgPath)

        findOrNot = True
        #predict
        for k in range(face_num_predict_):
            score = float(items1[2 + k * 5 + 4])
            #print score
            if score < thresh:
                continue
            x_min = int(items1[2 + k * 5 + 0])
            y_min = int(items1[2 + k * 5 + 1])
            x_max = int(items1[2 + k * 5 + 2])
            y_max = int(items1[2 + k * 5 + 3])
            cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0),2)
            #label
            match_or_not = False
            for j in range(face_num_label):
                x_min_ = int(items0[2+15*j])
                y_min_ = int(items0[2+1+15*j])
                w = int(items0[2+2+15*j])
                h = int(items0[2+3+15*j])
                x_max_ = x_min_ + w
                y_max_ = y_min_ + h
                
                match_or_not_ = judge([x_min,y_min,x_max,y_max],[x_min_,y_min_,x_max_,y_max_],iou_thresh)
                if match_or_not_:
                    match_or_not = True
                    break
            if match_or_not == False:
                findOrNot = False
                break

        if not findOrNot  or (face_num_predict != face_num_label):
        #if  (face_num_predict < face_num_label):
            print( '预测人脸数目： ',face_num_predict)
            print( '标注人脸数目: ',face_num_label)
            print( items0[0])
            cnt += 1 
            #continue
            for k in range(face_num_label):
                x_min = int(items0[2 + k * 15 + 0]) 
                y_min = int(items0[2 + k * 15 + 1])
                w = int(items0[2 + k * 15 + 2])
                x_max = x_min + w
                h = int(items0[2 + k * 15 + 3])
                y_max = y_min + h
                x_min -= 5
                x_min = max(0,x_min)
                x_max += 5
                x_max = min(x_max,img.shape[1]-1)
                cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,0,255),2)
            cv2.imshow('img',img)
            if cv2.waitKey() == ord('s'):
            #if 1:
                outPath = os.path.join(imgOutDir,items0[0].split('/')[-1])
                print( 'save: ',outPath)
                cv2.imwrite(outPath,img)
    print( cnt)

def eval():
    '''
    评估在测试集上的表现
    '''

    #测试集标注文件： finenamne  faceNum  xmin0 ymin0 xmax0 ymax0  xmin1 ymin1 xmax1 ymax1 ......
    #label_file = '/nfs/project/faceData_InCar/test_label_new.txt'
    label_file = './2000_label.txt'
    predict_file = './predict/faceboxV1.1-2k-640-predict.txt'
    output_data_file = './eval_data.pkl'
    output_recall_far_file = './faceboxV1.1-2k-640-badcase-2k.txt'
    reverseImg_file = './reverseImg.txt'
    iou_thresh = 0.5


    labels = open(label_file,'r').readlines()
    predicts = open(predict_file,'r').readlines()

    labels = [label.strip() for label in labels]
    predicts = [predict.strip() for predict in predicts]
    # labels.sort()
    # predicts.sort()
    # assert len(labels) == len(predicts)

    reverseImgs = open(reverseImg_file,'r').readlines()
    reverseImgs = [line.strip() for line in reverseImgs]

    
    imgNames = []
    predict_bboxs = []
    scores = []
    match_or_nots = []
    face_nums = []
    total_face_nums = 0

    for i in range(len(labels)):

        # print( i,len(labels))
        items_label = labels[i].split(' ')
        # print(items_label)
        predicts[i] = str(predicts[i]).replace("  ", " ")
        items_predict = predicts[i].split(' ')

        img_name_label = items_label[0]
        img_name_predict = items_predict[0]

        name = img_name_label.split('/')[-1]
        # name = name.rsplit('_',1)[0]
        if name in reverseImgs:
            continue
        # print(name)
        assert name == img_name_predict
        face_num_label = int(items_label[1])
        # print(items_predict)
        if len(items_predict) == 1:
            face_num_predict = 0
        else:
            face_num_predict = int(float(items_predict[1]))
        total_face_nums += face_num_label
        for k in range(face_num_predict):
            # print(items_predict[2+k*5+0:2+k*5+5])
            x_min = int(items_predict[2+k*5+0])
            y_min = int(items_predict[2+k*5+1])
            x_max = int(items_predict[2+k*5+2])
            y_max = int(items_predict[2+k*5+3])
            score = float(items_predict[2+k*5+4])
            imgNames.append(img_name_label)
            scores.append(score)
            predict_bboxs.append([x_min,y_min,x_max,y_max])
            face_nums.append(face_num_label)
            match_or_not = False
            for j in range(face_num_label):
                x_min_ = int(items_label[2+15*j])
                y_min_ = int(items_label[2+1+15*j])
                w = int(items_label[2+2+15*j])
                h = int(items_label[2+3+15*j])
                x_max_ = x_min_ + w
                y_max_ = y_min_ + h

                #x_min_ = int(items_label[2+j*4+0])
                #y_min_ = int(items_label[2+j*4+1])
                #x_max_ = int(items_label[2+j*4+2])
                #y_max_ = int(items_label[2+j*4+3])
                match_or_not_ = judge([x_min,y_min,x_max,y_max],[x_min_,y_min_,x_max_,y_max_],iou_thresh)
                if match_or_not_:
                    match_or_not = True
                    #break
            # if 0:
                # if not match_or_not:
                #     #imgDir = '/Users/didi//workspace/project/FaceDetection/data/face_inCar/labelImgs/after_human_check/test/test_0'
                #     imgDir = '/nfs/project/'
                #     imgPath = os.path.join(imgDir,img_name_label)
                #     print( imgPath)
                #     img = cv2.imread(imgPath)
                #     cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0),1)
                #     cv2.imshow('img',img)
                #     cv2.waitKey()

            match_or_nots.append(match_or_not)

    #save data into pickle
    data = {'imgNames':imgNames,'predict_bboxs':predict_bboxs,'scores':scores,'match_or_nots':match_or_nots,'face_nums':face_nums,'total_face_nums':total_face_nums}
    fout = open(output_data_file,'wb')
    pickle.dump(data,fout)
    fout.close()

    #
    analysis(data, output_recall_far_file)

def dataAlignment():
    label_file = './test_label2_new.txt'
    predict_file = './predict_ret_hard.txt'
    f_predicts = open(predict_file, "r").readlines()
    f_labels = open(label_file, "r").readlines()
    f_labels1 = open('./test_label2_new1.txt', "w")
    predicts = [predict.strip() for predict in f_predicts]
    labels = [label.strip().split(" ") for label in f_labels]
    predicts.sort()
    labels.sort()
    pre_name = [str(predict).split(" ")[0] for predict in predicts]
    # print(pre_name)
    label_name = [str(label).split(" ")[0].split("/")[-1].replace("',", "") for label in labels]
    # print(label_name[1])
    print("dsffds" in pre_name)
    label_name = list(set(label_name))
    print(len(set(f_labels)))
    for i in range(len(label_name)):
        if label_name[i] in pre_name:
            f_labels1.write(f_labels[i])
        else:
            print("no e")
    f_labels1.close()


if __name__ ==  '__main__':
    eval()
    # dataAlignment()
    #checkBadCase(0.5)







