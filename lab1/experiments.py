"""Эксперименты лабораторной работы 1: влияние γ и функции вознаграждения."""
import json
import os
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_iteration import (ACTION_ARROWS, ACTION_NAMES, ENV_NAME, default_reward,
                         exact_value_iteration, format_policy, train)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
GAMMAS = [0.7, 0.9, 0.99]
SEEDS = list(range(10))
N_ITERS = 150

_env = gym.make(ENV_NAME).unwrapped
DESC = _env.desc.astype(str)
P = _env.P
N_S, N_A = _env.observation_space.n, _env.action_space.n
MAX_STEPS = 100  # TimeLimit среды FrozenLake 4x4


# ---------------- функции вознаграждения ----------------
def hole_penalty(s, a, s2, r, term, desc):
    """-1 за падение в полынью."""
    return -1.0 if desc.flat[s2] == "H" else r


def step_penalty_small(s, a, s2, r, term, desc):
    """-0.01 за каждый шаг, не ведущий к цели."""
    return r if r > 0 else -0.01


def step_penalty_big(s, a, s2, r, term, desc):
    """-0.2 за каждый шаг, не ведущий к цели (очень «дорогое» время)."""
    return r if r > 0 else -0.2


def hole_and_step(s, a, s2, r, term, desc):
    """-1 за полынью и -0.01 за шаг."""
    if desc.flat[s2] == "H":
        return -1.0
    return r if r > 0 else -0.01


REWARDS = {
    "default (G=+1)": default_reward,
    "hole=-1": hole_penalty,
    "step=-0.01": step_penalty_small,
    "step=-0.2": step_penalty_big,
    "hole=-1, step=-0.01": hole_and_step,
}


# ---------------- точная оценка стратегии по модели ----------------
def policy_stats(policy, horizon=MAX_STEPS):
    """Точные (по матрице переходов) вероятность дойти до цели за horizon шагов,
    вероятность упасть в полынью и ожидаемая длина эпизода."""
    dist = np.zeros(N_S); dist[0] = 1.0
    p_goal = p_hole = exp_len = 0.0
    for t in range(horizon):
        new = np.zeros(N_S)
        for s in np.nonzero(dist)[0]:
            for p, s2, r, term in P[s][policy[s]]:
                new[s2] += dist[s] * p
        exp_len += dist.sum()  # живых эпизодов на шаге t делают шаг
        for s in range(N_S):
            if DESC.flat[s] == "G":
                p_goal += new[s]; new[s] = 0
            elif DESC.flat[s] == "H":
                p_hole += new[s]; new[s] = 0
        dist = new
    return p_goal, p_hole, exp_len


def terminal_mask():
    return np.array([c in "HG" for c in DESC.flat])


def plot_policy(ax, Q, title):
    V = Q.max(axis=1).reshape(DESC.shape)
    pol = Q.argmax(axis=1)
    im = ax.imshow(V, cmap="viridis")
    for i in range(DESC.shape[0]):
        for j in range(DESC.shape[1]):
            s = i * DESC.shape[1] + j
            c = DESC[i, j]
            txt = c if c in "HG" else f"{ACTION_ARROWS[pol[s]]}\n{V[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if V[i, j] < V.max() * 0.6 else "black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    t0 = time.time()
    results = {}
    optimal_policies = {}
    for rname, rfn in REWARDS.items():
        for gamma in GAMMAS:
            key = f"{rname} | γ={gamma}"
            Q_exact = exact_value_iteration(gamma, rfn)
            pol_exact = Q_exact.argmax(axis=1)
            optimal_policies[key] = pol_exact
            curves, stats, agree, Qs = [], [], [], []
            for seed in SEEDS:
                agent, curve, solved = train(gamma, rfn, seed=seed,
                                             max_iters=N_ITERS, early_stop=False)
                curves.append(curve)
                pol = agent.policy()
                stats.append(policy_stats(pol))
                nt = ~terminal_mask()
                agree.append(float(np.mean(pol[nt] == pol_exact[nt])))
                Qs.append(agent.q_matrix())
            curves = np.array(curves)
            stats = np.array(stats)
            results[key] = dict(
                reward=rname, gamma=gamma, curves=curves,
                p_goal=stats[:, 0].mean(), p_goal_std=stats[:, 0].std(),
                p_hole=stats[:, 1].mean(), ep_len=stats[:, 2].mean(),
                agree=float(np.mean(agree)),
                exact=policy_stats(pol_exact), Q_exact=Q_exact,
                Q_learned=Qs[0], policy_learned=Qs[0].argmax(axis=1),
            )
            print(f"{key:28s} P(goal)={results[key]['p_goal']:.3f}±{results[key]['p_goal_std']:.3f} "
                  f"P(hole)={results[key]['p_hole']:.3f} len={results[key]['ep_len']:.1f} "
                  f"совп.с оптим.={results[key]['agree']:.2f} "
                  f"оптимум: P(goal)={results[key]['exact'][0]:.3f} len={results[key]['exact'][2]:.1f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---- число итераций до "Solved" (критерий методички) ----
    solve = {}
    for gamma in GAMMAS:
        its = []
        for seed in range(30):
            _, _, solved = train(gamma, default_reward, seed=seed, max_iters=500)
            its.append(solved if solved else np.nan)
        solve[gamma] = its
        print(f"γ={gamma}: решено за {np.nanmean(its):.1f}±{np.nanstd(its):.1f} итераций "
              f"(медиана {np.nanmedian(its):.0f}, не решено {np.isnan(its).sum()}/30)", flush=True)

    # ---- графики ----
    def smooth(x, k=10):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, len(REWARDS), figsize=(4.2 * len(REWARDS), 4), sharey=True)
    for ax, rname in zip(axes, REWARDS):
        for gamma in GAMMAS:
            c = results[f"{rname} | γ={gamma}"]["curves"].mean(axis=0)
            ax.plot(np.arange(len(smooth(c))) + 10, smooth(c), label=f"γ={gamma}")
        ax.axhline(0.8, ls="--", c="gray", lw=0.8)
        ax.set_title(f"Награда: {rname}"); ax.set_xlabel("итерация")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("доля успешных тестовых эпизодов\n(ср. по 10 запускам, скольз. окно 10)")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=120)

    # кривая в стиле методички (один запуск, без сглаживания)
    fig, ax = plt.subplots(figsize=(7, 4))
    for gamma in GAMMAS:
        ax.plot(results[f"default (G=+1) | γ={gamma}"]["curves"][0], label=f"γ={gamma}", lw=1)
    ax.set_xlabel("итерация"); ax.set_ylabel("reward (20 тестовых эпизодов)")
    ax.set_title("Кривые обучения, исходная награда, seed=0"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "learning_curves_raw.png"), dpi=120)

    fig, axes = plt.subplots(len(REWARDS), len(GAMMAS), figsize=(4.3 * len(GAMMAS), 3.8 * len(REWARDS)))
    for i, rname in enumerate(REWARDS):
        for j, gamma in enumerate(GAMMAS):
            r = results[f"{rname} | γ={gamma}"]
            plot_policy(axes[i, j], r["Q_learned"],
                        f"{rname}, γ={gamma}\nP(goal)={r['p_goal']:.2f}, длина={r['ep_len']:.1f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "policies.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([np.array(solve[g])[~np.isnan(solve[g])] for g in GAMMAS],
               tick_labels=[f"γ={g}" for g in GAMMAS])
    ax.set_ylabel("итераций до reward > 0.8"); ax.grid(alpha=0.3)
    ax.set_title("Скорость решения (30 запусков, исходная награда)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "iterations_to_solve.png"), dpi=120)

    # ---- сохранение сводки ----
    summary = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in v.items() if kk not in ("curves", "Q_exact", "Q_learned")}
               for k, v in results.items()}
    for k, v in summary.items():
        v["policy_learned"] = format_policy(results[k]["policy_learned"], DESC)
        v["policy_optimal"] = format_policy(optimal_policies[k], DESC)
        v["final_test_reward_last20"] = float(results[k]["curves"][:, -20:].mean())
    summary["iterations_to_solve"] = {str(g): [None if np.isnan(x) else int(x) for x in v]
                                      for g, v in solve.items()}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Готово за {time.time()-t0:.0f} с")


if __name__ == "__main__":
    main()
