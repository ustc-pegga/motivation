device = "redmi"
python edge.py \
    --type mbv1_input \
    --data_root ../paper \
    --device ${device}_npu

python edge.py \
    --type mbv1_input \
    --data_root ../paper \
    --device ${device}_cpu

python edge.py \
    --type mbv1_input \
    --data_root ../paper \
    --device ${device}_gpu