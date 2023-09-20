python export.py \
    --model mobilenetv1 \
    --dataset cifar10 \
    --type mbv1_input \
    --data_root ../paper

python export.py \
    --model mobilenetv1 \
    --dataset imagenet \
    --type mbv1_input \
    --data_root ../paper

python export.py \
    --model mobilenetv1 \
    --dataset tiny-imagenet \
    --type mbv1_input \
    --data_root ../paper

python export.py \
    --model mobilenetv1 \
    --dataset "128test" \
    --type mbv1_input \
    --data_root ../paper