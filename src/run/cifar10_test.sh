device = "mate30e"
python edge.py \
    --type cifar10_device \
    --data_root ../paper \
    --device mate30e_cpu

python edge.py \
    --type cifar10_device \
    --data_root ../paper \
    --device mate30e_gpu

python edge.py \
    --type cifar10_device \
    --data_root ../paper \
    --device mate30e_npu