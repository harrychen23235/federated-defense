#!/usr/bin/env bash
echo "start running perturbation for tiny-imagenet"
for i in {2..10..2}
do
    echo "currently running norm cap $i"
    python src/federated.py --data=tiny-imagenet --local_ep=2 --bs=256 --num_agents=20 --rounds=15 --partition='homo' \
            --num_corrupt=4  --client_lr=0.001 --poison_lr=0.001 --attack_mode='fixed_generator' --poison_frac=0.3 --malicious_style=mixed  \
            --attack_start_round=5  --load_pretrained=True --pretrained_path='../data/saved_models/tiny_64_pretrain/tiny-resnet.epoch_20' \
            --storing_dir='./perturbation_init_norm_seperate_vector_4_mali/tiny_perturbation_'$i --pattern_type="pixel" --save_checkpoint=True \
            --seperate_vector=True --save_trigger=True --norm_cap=$i --generator_lr=0.1
done
