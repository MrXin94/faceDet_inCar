# encoding:utf-8
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MultiBoxLoss(nn.Module):
    num_classes = 2

    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.focalloss = FocalLoss(gamma=2.0, size_average=True)

    def cross_entropy_loss(self, x, y):
        x = x.detach()
        y = y.detach()
        xmax = x.data.max()
        # xmax = xmax.detach()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1, keepdim=True)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1, 1))

    def wing_loss(self, landmarks, labels, w=10.0, epsilon=2.0):
        x = landmarks - labels
        onet = torch.autograd.Variable(torch.from_numpy(np.array([1.0])).type(torch.FloatTensor),
                                       requires_grad=False).cuda()
        # one = torch.autograd.Variable(torch.from_numpy(np.array([1.0])).type(torch.FloatTensor),
        #                               requires_grad=False)
        # print 'diff: ',x
        c = w * (1.0 - torch.log(onet + w / epsilon))
        absolute_x = torch.abs(x)
        cond = (w > absolute_x).float()
        non_linear_part = w * torch.log(1.0 + absolute_x / epsilon)
        linear_part = absolute_x - c
        losses = cond * non_linear_part + (1 - cond) * linear_part
        # print 'losses: ',losses
        loss = torch.mean(losses)
        return loss

    def hard_negative_mining(self, conf_loss, pos):
        '''
        conf_loss [N*21482,]
        pos [N,21482]
        return negative indice
        '''
        batch_size, num_boxes = pos.size()
        conf_loss[pos.view(-1, 1)] = 0  # 去掉正样本,the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)

        _, idx = conf_loss.sort(1, descending=True)
        _, rank = idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)

        neg = rank < num_neg.expand_as(rank)
        return neg

    def forward(self, loc_preds, loc_targets, loc_landmarks_preds, loc_landmarks_targets, conf_preds, conf_targets):
        '''
        loc_preds[batch,21842,4]
        loc_targets[batch,21842,4]
        conf_preds[batch,21842,2]
        conf_targets[batch,21842]
        '''
        batch_size, num_boxes, _ = loc_preds.size()
        # print(batch_size,num_boxes)
        # print('ok1')
        pos = conf_targets > 0  # 大于0的地方，说明匹配到了人脸框
        num_pos = pos.long().sum(1, keepdim=True)
        # print(torch.sum(pos))
        # print(conf_targets.size())
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.Tensor([0]), requires_grad=True)
        # print('ok2')
        pos_mask1 = pos.unsqueeze(2).expand_as(loc_preds)
        # print(pos_mask1.size())
        # print('pos_mask1 sum {}'.format(torch.sum(pos_mask1)))
        pos_loc_preds = loc_preds[pos_mask1].view(-1, 4)
        pos_loc_targets = loc_targets[pos_mask1].view(-1, 4)

        # landmarks
        if loc_landmarks_targets is not None and loc_landmarks_preds is not None:
            pos_mask2 = pos.unsqueeze(2).expand_as(loc_landmarks_preds)
            pos_loc_landmarks_preds = loc_landmarks_preds[pos_mask2].view(-1, 10)
            pos_loc_landmarks_targets = loc_landmarks_targets[pos_mask2].view(-1, 10)
            # landmarks loss
            # loc_landmarks_loss = F.smooth_l1_loss(pos_loc_landmarks_preds,pos_loc_landmarks_targets,size_average=False)
            loc_landmarks_loss = F.mse_loss(pos_loc_landmarks_preds, pos_loc_landmarks_targets, size_average=True)
        else:
            loc_landmarks_loss = None

        # bbox loss
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=True)


        # temp_conf_loss = Variable(requires_grad=False)
        # print('conf_loss size {}'.format(conf_loss.size()))
        N = num_pos.data.sum()
        N = N.to(torch.float32)
        if 1:
            conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes),
                                                conf_targets.view(-1, 1))
            neg = self.hard_negative_mining(conf_loss, pos)
            pos_mask = pos.unsqueeze(2).expand_as(conf_preds)

            neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
            # print('sum neg mask {} size {}'.format(torch.sum(neg_mask),neg_mask.size()))
            # print('sum pos mask {} size {}'.format(torch.sum(pos_mask),pos_mask.size()))
            # print(neg_mask)
            mask = (pos_mask + neg_mask).gt(0)
            # print('sum mask {} size {}'.format(torch.sum(mask),mask.size()))

            pos_and_neg = (pos + neg).gt(0)
            # print('sum neg {} size {}'.format(torch.sum(neg),neg.size()))
            # print('sum pos {}'.format(torch.sum(pos)))
            # print('sum pos_and_neg {}'.format(torch.sum(pos_and_neg)))
            # print('preds shape {}'.format(conf_preds.size()))
            preds = conf_preds[mask].view(-1, self.num_classes)
            targets = conf_targets[pos_and_neg]
            conf_loss = F.cross_entropy(preds, targets, size_average=False)
            N_ = pos_and_neg.data.sum()
            N_ = N_.to(torch.float32)
            conf_loss /= N_
        else:
            conf_loss = self.focalloss(conf_preds.view(-1, self.num_classes),
                                       conf_targets.view(-1, 1))

        # loc_loss /= N
        # loc_landmarks_loss /= N
        # conf_loss /= N
        # print (N,N_)
        # print('loc_loss:%f  loc_landmarks_loss: %f conf_loss:%f, pos_num:%d' % (loc_loss.data[0], loc_landmarks_loss.data[0],conf_loss.data[0], N))
        return loc_loss, loc_landmarks_loss, conf_loss
