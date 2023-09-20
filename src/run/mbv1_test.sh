CUDA_VISIBLE_DEVICES=0 \
python edge.py \
    --model mobilenetv1 \
    --dataset cifar10 \
    --type mbv1 \
    --data_root ../paper \
    --device mate30e_npu
