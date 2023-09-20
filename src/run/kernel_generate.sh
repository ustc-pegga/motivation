device="mate30e"
python dwkernel.py \
    --type kernel \
    --data_root ../paper \
    --device ${device}_cpu \
    --op DWConv \
    --input 112 \
    --cin 32 \
    --cout 32 \
    --stride 1 \
    --kernel 3 