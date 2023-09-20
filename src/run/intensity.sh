device="mate30e"
python edge.py \
    --type 112_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_cpu

python edge.py \
    --type 112_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_gpu

python edge.py \
    --type 112_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_npu

python edge.py \
    --type 56_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_cpu

python edge.py \
    --type 56_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_gpu

python edge.py \
    --type 56_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_npu

python edge.py \
    --type 28_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_cpu

python edge.py \
    --type 28_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_gpu

python edge.py \
    --type 28_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_npu


python edge.py \
    --type 14_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_cpu

python edge.py \
    --type 14_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_gpu

python edge.py \
    --type 14_DWConv_kernel \
    --data_root ../paper \
    --device ${device}_npu