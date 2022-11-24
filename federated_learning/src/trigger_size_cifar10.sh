#!/usr/bin/env bash
echo "start running trigger size for cifar10"
for i in {1..32..4}
do
    echo "currently running trigger size $i"
    python src/federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=10 --rounds=20 --partition='homo' \
            --num_corrupt=4  --client_lr=0.1 --attack_mode='normal' --poison_frac=0.1 --malicious_style=mixed  \
            --attack_start_round=10  --load_pretrained=True --pretrained_path='../data/saved_models/cifar_pretrain/model_last.pt.tar.epoch_200' \
            --storing_dir='./trigger_generation/cifar10_pattern_size_'$i --pattern_size=$i --pattern_type="size_test"

done
