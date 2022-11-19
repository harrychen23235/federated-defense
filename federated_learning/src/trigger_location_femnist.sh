#!/usr/bin/env bash
echo "start running trigger location for femnist"
for i in {0..24..3}
do
    for j in {0..24..3}
    do  
    echo "currently running location $i $j"
    python src/federated.py --data=fedemnist --load_pretrained=False --local_ep=2 --bs=256 --num_agents=10 --rounds=20 --partition='homo' \
        --num_corrupt=4  --client_lr=0.1 --poison_lr=0.05 --attack_mode='normal' --poison_frac=0.2 --malicious_style=mixed  \
        --attack_start_round=10 --storing_dir='./femnist_pattern_location_'$i'_'$j  --pattern_type="location_test" --pattern_location $i $j
    done
done
