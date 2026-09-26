"""Анализ результатов: кривые обучения, сводная таблица, визуализация стратегий."""
import csv
import glob
import json
import os

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import Agent, Config

RUNS = "runs"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
SOLVED = {"LunarLander-v3": 200, "CartPole-v1": 475, "Acrobot-v1": -100, "MountainCar-v0": -110}


def load(name):
    with open(os.path.join(RUNS, name, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    h = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    s = json.load(open(os.path.join(RUNS, name, "summary.json")))
    c = json.load(open(os.path.join(RUNS, name, "config.json")))
    return h, s, c


def load_agent(name, which="best"):
    _, _, c = load(name)
    cfg = Config(**{**c, "device": "cpu"})
    env = gym.make(cfg.env)
    agent = Agent(env, cfg)
    path = os.path.join(RUNS, name, f"q_{which}.pt")
    if not os.path.exists(path):
        path = os.path.join(RUNS, name, "q_final.pt")
    agent.Q.load_state_dict(torch.load(path, map_location="cpu"))
    return agent, cfg


def rollout(agent, env_name, seed, frames_every=None):
    env = gym.make(env_name, render_mode="rgb_array" if frames_every else None)
    s, _ = env.reset(seed=seed)
    total, t, frames, traj, actions = 0.0, 0, [], [s], []
    while True:
        if frames_every and t % frames_every == 0:
            frames.append(env.render())
        a = agent.act(s)
        s, r, term, trunc, _ = env.step(a)
        total += r; t += 1
        traj.append(s); actions.append(a)
        if term or trunc:
            if frames_every:
                frames.append(env.render())
            break
    env.close()
    return total, t, frames, np.array(traj), np.array(actions)


def fig_learning(names, title, fname, env="LunarLander-v3"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in names:
        h, _, _ = load(n)
        ax.plot(h["episode"], h["reward"], lw=0.4, alpha=0.35)
        ax.plot(h["episode"], h["avg100"], lw=1.8, label=n, color=ax.lines[-1].get_color())
    ax.axhline(SOLVED[env], ls="--", c="gray", lw=1)
    lo = min(SOLVED[env] - 400, -400) if env == "LunarLander-v3" else None
    if lo: ax.set_ylim(lo, 320)
    ax.set_xlabel("эпизод"); ax.set_ylabel("награда за эпизод (тонко) / среднее за 100 (жирно)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=110); plt.close(fig)


def montage(name, seeds, fname, which="best"):
    """Доп. задание 1: несколько эпизодов на разных конфигурациях среды.
    Кадры эпизода накладываются друг на друга (траектория модуля) + GIF."""
    agent, cfg = load_agent(name, which)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3.2))
    for ax, seed in zip(axes, seeds):
        total, t, frames, _, _ = rollout(agent, cfg.env, seed, frames_every=1)
        sel = frames[::max(1, len(frames) // 12)] + [frames[-1]]
        stack = np.stack(sel).astype(float)
        img = stack.min(axis=0) * 0.6 + stack[-1] * 0.4  # «стробоскоп»: тёмный модуль на светлом фоне не нужен —
        img = np.where(stack.std(axis=0) > 1, stack.max(axis=0), stack[-1])  # берём яркие пиксели модуля
        ax.imshow(img.astype(np.uint8)); ax.axis("off")
        ax.set_title(f"seed={seed}: R={total:.0f}, {t} шагов", fontsize=9)
        imageio.mimsave(os.path.join(OUT, f"{fname}_seed{seed}.gif"), frames[::2], duration=0.04, loop=0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname + ".png"), dpi=100); plt.close(fig)


def progress_montage(name, fname, seed=101):
    """Аналог рисунка 3 методички: поведение на одном и том же seed по ходу обучения
    (последний кадр эпизода, записанного при обучении)."""
    vids = sorted(glob.glob(os.path.join(RUNS, name, "recording", "episodes_trained_*", "*.mp4")),
                  key=lambda p: int(p.split("episodes_trained_")[1].split("/")[0]))
    pick = [v for v in vids if int(v.split("episodes_trained_")[1].split("/")[0]) in (0, 100, 200, 300, 500, 1000, 1999)]
    fig, axes = plt.subplots(1, len(pick), figsize=(3.2 * len(pick), 2.6))
    for ax, v in zip(axes, pick):
        frames = [f for f in imageio.get_reader(v)]
        stack = np.stack(frames[::max(1, len(frames) // 12)] + [frames[-1]]).astype(float)
        img = np.where(stack.std(axis=0) > 1, stack.max(axis=0), stack[-1])
        ax.imshow(img.astype(np.uint8)); ax.axis("off")
        ax.set_title(f"после {v.split('episodes_trained_')[1].split('/')[0]} эп.", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=100); plt.close(fig)


def strategy_analysis(name, seeds=range(20)):
    """Статистика по действиям и фазам посадки выученной стратегии."""
    agent, cfg = load_agent(name)
    counts = np.zeros(4); rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in seeds:
        total, t, _, traj, acts = rollout(agent, cfg.env, 20_000 + seed)
        counts += np.bincount(acts, minlength=4)
        rows.append((total, t))
        axes[0].plot(traj[:, 0], traj[:, 1], lw=0.8)
        if seed < 3:
            axes[1].plot(traj[1:, 1], label=f"y, seed {seed}")
            axes[1].plot(traj[1:, 3], ls="--", label=f"vy, seed {seed}")
    axes[0].axvspan(-0.2, 0.2, ymax=0.05, color="gold", alpha=0.6)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_title("Траектории (20 эпизодов)")
    axes[1].set_xlabel("шаг"); axes[1].set_title("Высота y и вертикальная скорость vy"); axes[1].legend(fontsize=7)
    for a in axes: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=110); plt.close(fig)
    frac = counts / counts.sum()
    return dict(action_frac=dict(zip(["noop", "left_engine", "main_engine", "right_engine"], frac.round(3).tolist())),
                mean_len=float(np.mean([r[1] for r in rows])), mean_reward=float(np.mean([r[0] for r in rows])))


def main():
    names = sorted(d for d in os.listdir(RUNS) if os.path.isfile(os.path.join(RUNS, d, "summary.json")))
    table = []
    for n in names:
        h, s, c = load(n)
        thr = SOLVED[c["env"]]
        first = next((int(e) for e, a in zip(h["episode"], h["avg100"]) if a >= thr and e >= 99), None)
        table.append(dict(name=n, env=c["env"], episodes=int(c["episodes"]),
                          first_solved=first,
                          steps_to_solve=int(h["total_steps"][first]) if first is not None else None,
                          best_avg100=round(float(h["avg100"][99:].max()), 1) if len(h["avg100"]) > 99 else None,
                          last_avg100=round(float(h["avg100"][-1]), 1),
                          eval_best=round(s.get("eval_best_mean", float("nan")), 1),
                          eval_best_succ=s.get("eval_best_success"),
                          eval_final=round(s["eval_final_mean"], 1),
                          eval_final_succ=s["eval_final_success"],
                          time_min=round(s["train_time_s"] / 60, 1)))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    for r in table:
        print(r)

    ll = [n for n in names if load(n)[2]["env"] == "LunarLander-v3"]
    groups = sorted({n.rsplit("_s", 1)[0] for n in ll})
    for g in groups:
        fig_learning([n for n in ll if n.rsplit("_s", 1)[0] == g], f"LunarLander, {g}", f"curve_{g}.png")
    # сравнение средних за 100 по конфигурациям (ср. по сидам, первые 1000 эпизодов)
    fig, ax = plt.subplots(figsize=(9, 5))
    for g in groups:
        hs = [load(n)[0]["avg100"][:1000] for n in ll if n.rsplit("_s", 1)[0] == g]
        m = min(map(len, hs)); arr = np.array([x[:m] for x in hs])
        l, = ax.plot(arr.mean(0), label=g, lw=1.8)
        ax.fill_between(np.arange(m), arr.min(0), arr.max(0), color=l.get_color(), alpha=0.15)
    ax.axhline(200, ls="--", c="gray"); ax.set_ylim(-300, 300)
    ax.set_xlabel("эпизод"); ax.set_ylabel("средняя награда за 100 эпизодов (ср. по 2 сидам, заливка — min/max)")
    ax.set_title("LunarLander: влияние гиперпараметров"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "hyperparams.png"), dpi=110); plt.close(fig)

    for n in names:
        env = load(n)[2]["env"]
        if env != "LunarLander-v3":
            fig_learning([n], env, f"curve_{n}.png", env)

    if "baseline_s0" in names:
        progress_montage("baseline_s0", "progress_baseline.png")
    best = max((r for r in table if r["env"] == "LunarLander-v3"), key=lambda r: r["eval_best"])
    print("best run:", best["name"])
    montage(best["name"], [1, 2, 3, 4], "landings_best")
    info = strategy_analysis(best["name"])
    info["run"] = best["name"]
    print(info)
    json.dump(info, open(os.path.join(OUT, "strategy.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
