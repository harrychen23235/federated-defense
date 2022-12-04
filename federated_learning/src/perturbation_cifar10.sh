#!/usr/bin/env bash
echo "start running perturbation for cifar10"
for i in {1..9..1}
do
    echo "currently running norm cap $i"
    python src/federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=20 --rounds=8 --partition='homo' \
            --num_corrupt=4  --client_lr=0.1 --attack_mode='fixed_generator' --poison_frac=0.1 --malicious_style=mixed  \
            --attack_start_round=3  --load_pretrained=True --pretrained_path='../data/saved_models/cifar_pretrain/model_last.pt.tar.epoch_200' \
            --storing_dir='./perturbation_init_norm_seperate_vector_4_mali/cifar10_perturbation_0_'$i --pattern_type="pixel" \
            --alpha=0.8 --save_checkpoint=True --seperate_vector=True --save_trigger=True --norm_cap='0.'$i --generator_lr=0.1
done
