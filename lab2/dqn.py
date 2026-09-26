"""Лабораторная работа 2: глубокое Q-обучение (DQN) для посадки лунного модуля.

Структура повторяет пример из методички (Q_network, ReplayBuffer, Agent,
цикл обучения, visualize_episode), гиперпараметры по умолчанию — из методички.
Все параметры собраны в Config, чтобы запускать эксперименты из командной строки:

    python dqn.py                                   # базовый запуск (как в методичке)
    python dqn.py --name fast --lr 5e-4 --sync-mode steps --sync-iterations 1000
    python dqn.py --env CartPole-v1 --episodes 500
"""
import argparse
import csv
import dataclasses
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class Config:
    name: str = "baseline"
    env: str = "LunarLander-v3"
    episodes: int = 2000            # LAST_EPISODE
    sync_iterations: int = 2        # SYNC_ITERATIONS
    sync_mode: str = "episode"      # "episode" — как в методичке, "steps" — каждые sync_iterations шагов обучения
    lr: float = 1e-4                # LEARNING_RATE
    gamma: float = 0.99             # GAMMA
    epsilon_decay: float = 0.99     # EPSILON_DECAY (умножение после каждого эпизода)
    epsilon_final: float = 0.02     # EPSILON_FINAL
    batch_size: int = 64            # BATCH_SIZE
    buffer_size: int = 65536        # REPLAY_BUFFER_SIZE
    hidden1: int = 256              # HIDDEN_LAYER_SIZE1
    hidden2: int = 128              # HIDDEN_LAYER_SIZE2
    train_interval: int = 1         # TRAIN_INTERVAL
    optimizer: str = "rmsprop"      # "rmsprop" (методичка) или "adam"
    loss: str = "mse"               # "mse" или "huber"
    double_dqn: bool = False
    done_on_truncation: bool = False  # True — как в методичке (обрыв по лимиту времени считается концом эпизода)
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs"
    eval_episodes: int = 100
    record_every: int = 100         # визуализация каждые N эпизодов (0 — не записывать)


class Q_network(nn.Module):
    def __init__(self, action_n, input_s, hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_s, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_n),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Кольцевой буфер повторов на тензорах PyTorch (как в методичке)."""

    def __init__(self, size, state_dim, device="cpu"):
        self.size, self.device = size, device
        self.ptr, self.count = 0, 0
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, 1), dtype=torch.int64, device=device)
        self.next_states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size, 1), dtype=torch.bool, device=device)

    def append(self, state, action, next_state, reward, done):
        idx = self.ptr
        self.states[idx] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[idx] = int(action)
        self.next_states[idx] = torch.as_tensor(next_state, dtype=torch.float32)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.count, (batch_size,), device=self.device)
        return (self.states[indices], self.actions[indices], self.next_states[indices],
                self.rewards[indices], self.dones[indices])

    def __len__(self):
        return self.count


class Agent:
    def __init__(self, train_env, cfg: Config):
        self.cfg = cfg
        n_actions = train_env.action_space.n
        obs_dim = int(np.prod(train_env.observation_space.shape))
        self.Q = Q_network(n_actions, obs_dim, cfg.hidden1, cfg.hidden2).to(cfg.device)
        self.target_Net = Q_network(n_actions, obs_dim, cfg.hidden1, cfg.hidden2).to(cfg.device)
        if cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=cfg.lr)
        else:
            self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=cfg.lr, weight_decay=1e-4)
        self.target_Net.load_state_dict(self.Q.state_dict())
        self.replay_buffer = ReplayBuffer(cfg.buffer_size, obs_dim, torch.device(cfg.device))
        self.loss_fn = nn.SmoothL1Loss() if cfg.loss == "huber" else nn.MSELoss()

    def targetNetupdate(self):
        self.target_Net.load_state_dict(self.Q.state_dict())

    @torch.no_grad()
    def act(self, state):
        q_values = self.Q(torch.as_tensor(state, dtype=torch.float32, device=self.cfg.device))
        return q_values.argmax().item()

    def train_iteration(self):
        if len(self.replay_buffer) < self.cfg.batch_size:
            return 0.0
        self.optimizer.zero_grad()
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.cfg.batch_size)

        # ---- фрагмент, который требуется реализовать по заданию ----
        # Q(s_t, a_t; θ) — ценность выполненного действия по основной сети
        action_values = self.Q(states)                          # [B, n_actions]
        Qs = action_values.gather(index=actions, dim=1)         # [B, 1]
        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN: действие выбирает основная сеть, оценивает — целевая
                next_actions = self.Q(next_states).argmax(dim=1, keepdim=True)
                Q_next_max = self.target_Net(next_states).gather(1, next_actions)
            else:
                # max_a' Q(s_{t+1}, a'; θ⁻) — по целевой сети
                Q_next_max = self.target_Net(next_states).max(dim=1, keepdim=True)[0]
            Q_next_max[dones] = 0.0                              # конец эпизода: ценность будущего = 0
            target = rewards + self.cfg.gamma * Q_next_max       # правая часть уравнения Беллмана
        loss = self.loss_fn(Qs, target)                         # (r + γ·max Q⁻ − Q)²
        # -------------------------------------------------------------

        loss.backward()
        self.optimizer.step()
        return loss.item()


def evaluate(agent, env_name, episodes=100, seed_base=10_000):
    """Жадная стратегия на `episodes` случайных конфигурациях среды."""
    env = gym.make(env_name)
    rewards = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed_base + i)
        total = 0.0
        while True:
            state, r, term, trunc, _ = env.step(agent.act(state))
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()
    return np.array(rewards)


def visualize_episode(agent, env_name, video_folder, seeds=(101,), episodes_trained=None):
    """Запись эпизодов обученного агента в видео (RecordVideo), по одному на каждый seed.
    В отличие от методички, записываются несколько разных конфигураций среды (доп. задание 1)."""
    from gymnasium.wrappers import RecordVideo
    if episodes_trained is not None:
        video_folder = os.path.join(video_folder, f"episodes_trained_{episodes_trained}")
    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(gym.make(env_name, render_mode="rgb_array"), video_folder=video_folder,
                      episode_trigger=lambda ep: True, disable_logger=True)
    results = []
    for seed in seeds:
        state, _ = env.reset(seed=int(seed))
        total = 0.0
        while True:
            state, r, term, trunc, _ = env.step(agent.act(state))
            total += r
            if term or trunc:
                break
        results.append(total)
    env.close()
    return results


def train(cfg: Config, verbose=True):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    train_env = gym.make(cfg.env)
    train_env.action_space.seed(cfg.seed)
    agent = Agent(train_env, cfg)
    epsilon = 1.0
    scores, history = [], []
    train_iterations = 0
    buffer_updates = 0
    total_steps = 0
    best_avg, best_state = -np.inf, None
    t_start = time.time()

    for episode in range(cfg.episodes):
        state, _ = train_env.reset(seed=cfg.seed * 100_000 + episode)
        total_reward, total_loss, episode_length = 0.0, 0.0, 0
        buffer_updates += 1
        while True:
            episode_length += 1
            if random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                action = agent.act(state)
            new_state, reward, terminated, truncated, _ = train_env.step(action)
            is_done = terminated or truncated
            # При обрыве эпизода по лимиту времени (truncated) состояние не терминальное,
            # поэтому по умолчанию в буфер пишем только terminated.
            done_flag = is_done if cfg.done_on_truncation else terminated
            agent.replay_buffer.append(state, action, new_state, reward, done_flag)
            state = new_state
            total_reward += reward
            if len(agent.replay_buffer) > cfg.batch_size and buffer_updates % cfg.train_interval == 0:
                total_loss += agent.train_iteration()
                train_iterations += 1
                if cfg.sync_mode == "steps" and train_iterations % cfg.sync_iterations == 0:
                    agent.targetNetupdate()
            if is_done:
                break
        total_steps += episode_length
        if cfg.sync_mode == "episode" and train_iterations % cfg.sync_iterations == 0 and train_iterations > 0:
            agent.targetNetupdate()
        epsilon = max(cfg.epsilon_final, epsilon * cfg.epsilon_decay)

        scores.append(total_reward)
        avg100 = float(np.mean(scores[-100:]))
        history.append(dict(episode=episode, reward=total_reward, avg100=avg100, steps=episode_length,
                            total_steps=total_steps, loss=total_loss / max(episode_length, 1),
                            epsilon=epsilon, time=time.time() - t_start))
        if len(scores) >= 100 and avg100 > best_avg:
            best_avg = avg100
            best_state = {k: v.clone() for k, v in agent.Q.state_dict().items()}
        if verbose and (episode % 20 == 0 or episode == cfg.episodes - 1):
            print(f"[{cfg.name}] Episode {episode}, Steps: {episode_length}, Reward: {total_reward:.2f}, "
                  f"Avg100: {avg100:.2f}, Epsilon: {epsilon:.2f}, Buffer: {len(agent.replay_buffer)}, "
                  f"t={time.time() - t_start:.0f}s", flush=True)
        if cfg.record_every and (episode % cfg.record_every == 0 or episode == cfg.episodes - 1):
            visualize_episode(agent, cfg.env, os.path.join(run_dir, "recording"),
                              seeds=(101,), episodes_trained=episode)

    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    torch.save(agent.Q.state_dict(), os.path.join(run_dir, "q_final.pt"))
    if best_state is not None:
        torch.save(best_state, os.path.join(run_dir, "q_best.pt"))

    # Итоговая оценка жадной стратегии на 100 новых конфигурациях среды
    final_eval = evaluate(agent, cfg.env, cfg.eval_episodes)
    summary = dict(name=cfg.name, env=cfg.env, episodes=cfg.episodes, total_steps=total_steps,
                   train_time_s=time.time() - t_start,
                   first_avg100_ge_200=next((h["episode"] for h in history if h["avg100"] >= 200
                                             and h["episode"] >= 99), None),
                   best_avg100=best_avg if best_state is not None else None, last_avg100=history[-1]["avg100"],
                   eval_final_mean=float(final_eval.mean()), eval_final_std=float(final_eval.std()),
                   eval_final_success=float((final_eval >= 200).mean()))
    if best_state is not None:
        agent.Q.load_state_dict(best_state)
        best_eval = evaluate(agent, cfg.env, cfg.eval_episodes)
        summary.update(eval_best_mean=float(best_eval.mean()), eval_best_std=float(best_eval.std()),
                       eval_best_success=float((best_eval >= 200).mean()))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(json.dumps(summary, indent=2))
    return agent, history, summary


def parse_args():
    p = argparse.ArgumentParser()
    for field in dataclasses.fields(Config):
        flag = "--" + field.name.replace("_", "-")
        if field.type is bool or field.type == "bool":
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=field.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(field.type, field.type) \
                if isinstance(field.type, str) else field.type
            p.add_argument(flag, type=typ, default=field.default)
    return Config(**vars(p.parse_args()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    train(parse_args())
