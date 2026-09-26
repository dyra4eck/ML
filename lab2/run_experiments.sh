#!/bin/bash
# Запуск всех экспериментов (4 процесса параллельно). Логи — runs/<name>.log
cd "$(dirname "$0")"
mkdir -p runs
cat <<'LIST' | xargs -P 4 -I{} bash -c 'set -- {}; name=$1; shift; python3 dqn.py --name $name "$@" > runs/$name.log 2>&1'
baseline_s0 --seed 0 --episodes 2000
baseline_s1 --seed 1 --episodes 2000
sync1000_s0 --seed 0 --episodes 1000 --sync-mode steps --sync-iterations 1000 --record-every 0
sync1000_s1 --seed 1 --episodes 1000 --sync-mode steps --sync-iterations 1000 --record-every 0
lr5e-4_s0 --seed 0 --episodes 1000 --lr 5e-4 --record-every 0
lr5e-4_s1 --seed 1 --episodes 1000 --lr 5e-4 --record-every 0
eps0.98_s0 --seed 0 --episodes 1000 --epsilon-decay 0.98 --record-every 0
eps0.98_s1 --seed 1 --episodes 1000 --epsilon-decay 0.98 --record-every 0
huber_s0 --seed 0 --episodes 1000 --loss huber --record-every 0
huber_s1 --seed 1 --episodes 1000 --loss huber --record-every 0
tuned_s0 --seed 0 --episodes 1000 --optimizer adam --lr 5e-4 --sync-mode steps --sync-iterations 500 --epsilon-decay 0.98 --loss huber --double-dqn true
tuned_s1 --seed 1 --episodes 1000 --optimizer adam --lr 5e-4 --sync-mode steps --sync-iterations 500 --epsilon-decay 0.98 --loss huber --double-dqn true
cartpole --env CartPole-v1 --episodes 500 --epsilon-decay 0.98 --record-every 100
acrobot --env Acrobot-v1 --episodes 300 --epsilon-decay 0.98 --record-every 100
mountaincar --env MountainCar-v0 --episodes 500 --epsilon-decay 0.98 --record-every 100
LIST
