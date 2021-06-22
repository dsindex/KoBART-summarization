python train.py \
    --train_file data_aihub/train_news.txt \
    --test_file data_aihub/dev_news.txt \
    --gradient_clip_val 1.0 \
    --max_epochs 5 \
    --default_root_dir logs  \
    --gpus 1 \
    --lr 3e-5 \
    --batch_size 16 \
    --num_workers 4
