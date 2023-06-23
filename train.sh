python finetune.py \
    --data_root ./dataset_name \
    --model_type vit_h \
    --checkpoint_path ../SAM_customizing/sam_vit_h.pth \
    --freeze_image_encoder \
    --freeze_mask_decoder \
    --batch_size 2 \
    --image_size 1024 \
    --steps 1500 \
    --learning_rate 1.e-5 \
    --weight_decay 0.01