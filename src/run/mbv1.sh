
python export.py \
    --model mobilenetv1 \
    --dataset cifar10 \
    --type mbv1 \
    --data_root ../paper \
    --rate "1.0, 0.75, 0.5, 0.9375, 1.0, 1.0, 0.875, 0.65625, 0.84375, 0.328125, 0.453125, 0.640625, 0.8125, 0.203125, 0.3828125" \
    --name "amc"


python export.py \
    --model mobilenetv1 \
    --dataset cifar100 \
    --type mbv1 \
    --data_root ../paper \
    --rate "1.0, 0.75, 0.75, 0.6875, 0.75, 0.6875, 0.71875, 0.6875, 0.6875, 0.6875, 0.671875, 0.671875, 0.734375, 0.7890625, 0.515625" \
    --name "amc"


python export.py \
    --model mobilenetv1 \
    --dataset cifar10 \
    --type mbv1 \
    --data_root ../paper \
    --kernel_list "5, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3" \
    --rate "1.0, 1.0, 0.9375, 0.875, 0.84375, 0.796875, 0.765625, 0.7109375, 0.6015625, 0.5078125, 0.484375, 0.4765625, 0.640625, 0.76171875, 0.3125" \
    --name "GPU_pruning"



python export.py \
    --model mobilenetv1 \
    --dataset cifar100 \
    --type mbv1 \
    --data_root ../paper \
    --kernel_list "5, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3" \
    --rate "1.0, 0.625, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.703125, 0.703125, 0.703125, 0.6953125, 0.6875, 0.6953125, 0.71484375, 0.69921875" \
    --name "GPU_pruning"


python export.py \
    --model mobilenetv1 \
    --dataset cifar10 \
    --type mbv1 \
    --data_root ../paper \
    --name "origin"


python export.py \
    --model mobilenetv1 \
    --dataset cifar100 \
    --type mbv1 \
    --data_root ../paper \
    --name "origin"

python export.py \
    --model mobilenetv1 \
    --dataset tiny-imagenet \
    --type mbv1 \
    --data_root ../paper \
    --rate "1.0, 0.75, 0.75, 0.625, 0.6875, 0.625, 0.6875, 0.671875, 0.6875, 0.6875, 0.6875, 0.703125, 0.765625, 0.84375, 0.640625" \
    --name "amc"

python export.py \
    --model mobilenetv1 \
    --dataset tiny-imagenet \
    --type mbv1 \
    --data_root ../paper \
    --kernel_list "5, 7, 5, 9, 5, 5, 3, 3, 3, 3, 3, 3, 3" \
    --rate "1.0, 0.875, 0.875, 0.84375, 0.78125, 0.71875, 0.671875, 0.6171875, 0.6015625, 0.59375, 0.6015625, 0.609375, 0.65625, 0.75390625, 0.5234375" \
    --name "GPU_pruning"

python export.py \
    --model mobilenetv1 \
    --dataset tiny-imagenet \
    --type mbv1 \
    --data_root ../paper \
    --kernel_list "5, 7, 5, 9, 5, 5, 3, 3, 3, 3, 3, 3, 3" \
    --name "GPU"