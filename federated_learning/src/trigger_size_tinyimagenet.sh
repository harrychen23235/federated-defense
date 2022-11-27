#!/usr/bin/env bash
echo "start running trigger size for tiny_imagenet"
for i in {1..64..5}
do
    echo "currently running trigger size $i"
    python src/federated.py --data=tiny-imagenet --local_ep=2 --bs=256 --num_agents=20 --rounds=10 --partition='homo' \
            --num_corrupt=4  --client_lr=0.1 --attack_mode='normal' --poison_frac=0.1 --malicious_style=mixed  \
            --attack_start_round=5  --load_pretrained=True --pretrained_path='../data/saved_models/cifar_pretrain/model_last.pt.tar.epoch_200' \
            --storing_dir='./trigger_size_imagenet/imagenet_pattern_size_'$i --pattern_size=$i --pattern_type="size_test" --save_checkpoint=True \

done
