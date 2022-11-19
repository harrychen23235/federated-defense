#!/usr/bin/env bash
echo "start running trigger size for femnist"
for i in {1..28..3}
do
    echo "currently running trigger size $i"
    python src/federated.py --data=fedemnist --load_pretrained=False --local_ep=2 --bs=256 --num_agents=10 --rounds=20 --partition='homo' \
        --num_corrupt=4  --client_lr=0.1 --poison_lr=0.05 --attack_mode='normal' --poison_frac=0.2 --malicious_style=mixed  \
        --attack_start_round=10 --storing_dir='./femnist_pattern_size_'$i --pattern_size=$i --pattern_type="size_test"
done
