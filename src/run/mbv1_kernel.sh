device="mate30e"
python edge.py \
    --type mobilenetv1_kernel \
    --data_root ../paper \
    --device ${device}_cpu

python edge.py \
    --type mobilenetv1_kernel \
    --data_root ../paper \
    --device ${device}_gpu

python edge.py \
    --type mobilenetv1_kernel \
    --data_root ../paper \
    --device ${device}_npu