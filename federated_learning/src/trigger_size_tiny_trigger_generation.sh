#!/usr/bin/env bash
echo "start running trigger size(trigger_generation) for tiny-imagenet"
for i in {1..64..5}
do
    echo "currently running trigger size $i"
    python src/federated.py --data=tiny-imagenet --local_ep=2 --bs=256 --num_agents=20 --rounds=15 --partition='homo' \
            --num_corrupt=4  --client_lr=0.001 --poison_lr=0.001 --attack_mode='fixed_generator' --poison_frac=0.3 --malicious_style=mixed  \
            --attack_start_round=5  --load_pretrained=True --pretrained_path='../data/saved_models/tiny_64_pretrain/tiny-resnet.epoch_20' \
            --storing_dir='./trigger_generation_seperate_vector_4_mali/tiny_pattern_size_trigger_generation_'$i --pattern_size=$i --pattern_type="size_test" --save_checkpoint=True \
            --seperate_vector=True --save_trigger=True
done
