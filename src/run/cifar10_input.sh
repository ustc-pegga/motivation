python export.py \
    --model mobilenetv1 \
    --dataset cifar10 \
    --type cifar10_device \
    --data_root ../paper

python export.py \
    --model mobilenetv2 \
    --dataset cifar10 \
    --type cifar10_device \
    --data_root ../paper

python export.py \
    --model resnet18 \
    --dataset cifar10 \
    --type cifar10_device \
    --data_root ../paper