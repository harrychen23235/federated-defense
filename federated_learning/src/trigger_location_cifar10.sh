#!/usr/bin/env bash
echo "start running trigger location for cifar10"
for i in {0..28..5}
do
    for j in {0..28..5}
    do  
    echo "currently running location $i $j"
    python src/federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=10 --rounds=15 --partition='homo' \
            --num_corrupt=4  --client_lr=0.1 --attack_mode='normal' --poison_frac=0.1 --malicious_style=mixed  \
            --attack_start_round=10  --load_pretrained=True --pretrained_path='../data/saved_models/cifar_pretrain/model_last.pt.tar.epoch_200' \
            --storing_dir='./cifar10_pattern_location_'$i'_'$j  --pattern_type="location_test" --pattern_location $i $j
    done
done
