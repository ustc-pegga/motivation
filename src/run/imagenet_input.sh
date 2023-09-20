python export.py \
    --model mobilenetv1 \
    --dataset imagenet \
    --type imagenet_device \
    --data_root ../paper

python export.py \
    --model mobilenetv2 \
    --dataset imagenet \
    --type imagenet_device \
    --data_root ../paper

python export.py \
    --model resnet18 \
    --dataset imagenet \
    --type imagenet_device \
    --data_root ../paper