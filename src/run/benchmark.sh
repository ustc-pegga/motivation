device = "mi11"
python edge.py \
    --type benchmark \
    --data_root ../paper \
    --device mi11_cpu

python edge.py \
    --type benchmark \
    --data_root ../paper \
    --device mi11_gpu

python edge.py \
    --type benchmark \
    --data_root ../paper \
    --device mi11_npu