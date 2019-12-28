# facebox优化

## Introduction

本工程是对facebox在车载场景中的优化

## Data

1. label目录为./widerface_2w_1p5w_3p2w_1p2w-label.txt, label文件格式为

```Shell
filename bbox_num bbox_x1 bbox_y1 bbox_x2 bbox_y2 landmark_x1 landmark_y1 landmark_x2 landmark_y2 landmark_x3 landmark_y3 landmark_x4 landmark_y4 landmark_x5 landmark_x6 blur
35--Basketball/35_Basketball_basketballgame_ball_35_34.jpg 2 576 0 93 51 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 350 44 29 38 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1
35--Basketball/35_Basketball_playingbasketball_35_441.jpg 1 390 144 102 159 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1
35--Basketball/35_Basketball_playingbasketball_35_204.jpg 4 834 291 45 49 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 618 316 32 30 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 145 325 50 75 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 592 445 32 38 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1
35--Basketball/35_Basketball_basketballgame_ball_35_495.jpg 3 902 775 42 65 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 497 417 82 110 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 955 536 36 45 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1
```
2. 可以选择lmdb数据或原始数据

## requirement

```Shell
future (0.18.2)
lmdb (0.98)
numpy (1.16.5)
opencv-python (4.1.2.30)
Pillow (6.2.1)
six (1.13.0)
torch (1.2.1)
torchvision (0.4.2)
```
## Training

``sh train.sh``

train.sh:
```Shell
python -u trainvisdom.py \
--network faceboxV2 \
--dataset_type lmdb \
--label widerface_2w_1p5w_3p2w_1p2w-label.txt \
--scale 1024 \
--lr 0.001 \
--batch 64 \
--start_epoch 209 \
--fineturn True \
--model_path weight/faceboxV1_rfb_wider_2w_1p5_3p2_1p2_80.pt \
>> log.txt&
```

## Testing

Please check ``test.py`` for testing.

## Pretrained Models list

模型                 | 数据集                                        | 分辨率 | 2k测试集   | 6k测试集   |
|------------------------|-----------------------------------------------|--------|---------|---------|
| facebox                | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 92\.99% | 96\.83% |
|                        |                                               | 1024   | 98\.74% | 97\.53% |
| facebox\_noBn        | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 86\.09% | 96\.42% |
|                        |                                               | 1024   | 98\.54% | 97\.52% |
| facebox\_noBn\_rfb   | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 85\.66% | 95\.91% |
|                        |                                               | 1024   | 98\.6%  | 97\.51% |
| faceboxV2              | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 88\.71% | 96\.47% |
|                        |                                               | 1024   | 98\.62% | 97\.74% |
| faceboxV2\_fused       | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 88\.97% | 96\.27% |
|                        |                                               | 1024   | 98\.52% | 97\.61% |
| faceboxV2\_noBn        | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 80\.85% | 94\.94% |
|                        |                                               | 1024   | 97\.77% | 97\.38% |
| faceboxV2\_noBn\_fused | widerface\+2w\+1\.5w\+3\.2w\(noCheck\)\+1\.2w | 640    | 80\.56% | 94\.7%  |
|                        |                                               | 1024   | 97\.66% | 97\.44% |

##fusion

模型融合借鉴于[ACNet](https://arxiv.org/abs/1908.03930) 的模型融合部分，faceboxV2和faceboxV2_noBn中生成faceboxV2_fusion和faceboxV2_noBn_fusion需要进行模型融合，可以使用脚本
``python acnet_fusion``转换




