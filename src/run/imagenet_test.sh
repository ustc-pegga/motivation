device = "redmi"
python edge.py \
    --type imagenet_device \
    --data_root ../paper \
    --device redmi_cpu

python edge.py \
    --type imagenet_device \
    --data_root ../paper \
    --device redmi_gpu

python edge.py \
    --type imagenet_device \
    --data_root ../paper \
    --device redmi_npu