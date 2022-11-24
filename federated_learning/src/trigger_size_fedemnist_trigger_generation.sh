#!/usr/bin/env bash
echo "start running trigger size(trigger_generation) for femnist"
for i in {1..28..3}
do
    echo "currently running trigger size $i"
    python src/federated.py --data=fedemnist --local_ep=2 --bs=256 --num_agents=10 --rounds=20 --partition='homo' \
        --num_corrupt=2  --client_lr=0.1 --poison_lr=0.05 --attack_mode='fixed_generator' --poison_frac=0.2 --malicious_style=mixed  \
        --attack_start_round=5 --storing_dir='./femnist_pattern_size_trigger_generation_'$i --pattern_size=$i --pattern_type="size_test" \
        --load_pretrained=True --pretrained_path='../data/saved_models/femnist_pretrain/checkpoint_10.pt' --args.alpha=0.5


done
