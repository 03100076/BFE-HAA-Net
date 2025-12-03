OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BFE-HAA-Net.yaml \
    --eval-only \
    --num-gpus 2 \
    OUTPUT_DIR /home/BFE-HAA-Net/results \
    MODEL.WEIGHTS model_0099999.pth
