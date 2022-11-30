#!/usr/bin/env bash
echo "start running trigger size(trigger_generation) for cifar10"
for i in {1..32..4}
do
    echo "currently running trigger size $i"
    python src/federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=20 --rounds=15 --partition='homo' \
            --num_corrupt=4  --client_lr=0.1 --attack_mode='fixed_generator' --poison_frac=0.1 --malicious_style=mixed  \
            --attack_start_round=5  --load_pretrained=True --pretrained_path='../data/saved_models/cifar_pretrain/model_last.pt.tar.epoch_200' \
            --storing_dir='./trigger_generation_share_vector_4_mali/cifar10_trigger_generation_pattern_size_'$i --pattern_size=$i --pattern_type="size_test" \
            --alpha=0.8 --save_checkpoint=True
done
