python -u trainvisdom.py \
--network faceboxV2 \
--dataset_type lmdb \
--label widerface_2w_1p5w_3p2w-label.txt \
--scale 1024 \
--lr 0.001 \
--batch 64 \
--start_epoch 209 \
--fineturn True \
--model_path weight/faceboxV1_rfb_wider_2w_1p5_3p2_1p2_80.pt \
>> log.txt&
