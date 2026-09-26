"""Лабораторная работа 1: алгоритм итерации по ценностям действий (Q-iteration)
для среды FrozenLake.

Реализация следует методичке (Lapan, "Deep RL Hands-On", гл. 5), но использует
библиотеку gymnasium (преемник gym): среда FrozenLake-v0 в ней называется
FrozenLake-v1, а step() возвращает (obs, reward, terminated, truncated, info).
"""
import collections

import gymnasium as gym
import numpy as np

ENV_NAME = "FrozenLake-v1"
TEST_EPISODES = 20
ACTION_NAMES = ["Left", "Down", "Right", "Up"]
ACTION_ARROWS = ["←", "↓", "→", "↑"]


def default_reward(state, action, new_state, reward, terminated, desc):
    """Исходная функция вознаграждения среды: 1 за достижение цели, иначе 0."""
    return reward


class Agent:
    def __init__(self, gamma=0.9, reward_fn=default_reward, seed=None):
        self.gamma = gamma
        self.reward_fn = reward_fn
        self.env = gym.make(ENV_NAME)
        self.desc = self.env.unwrapped.desc.astype(str)
        self.state, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.rng = np.random.default_rng(seed)
        # (s, a, s') -> вознаграждение r
        self.reward_table = collections.defaultdict(float)
        # (s, a) -> Counter{s': число переходов}
        self.transition_table = collections.defaultdict(collections.Counter)
        # (s, a) -> Q(s, a)
        self.Q = collections.defaultdict(float)

    def _record(self, state, action, new_state, reward, terminated):
        shaped = self.reward_fn(state, action, new_state, reward, terminated, self.desc)
        self.reward_table[(state, action, new_state)] = shaped
        self.transition_table[(state, action)][new_state] += 1

    def play_n_random_steps(self, count):
        """Сбор статистики о среде случайными действиями."""
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            self._record(self.state, action, new_state, reward, terminated)
            if terminated or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def select_action(self, state):
        """Жадный выбор действия по Q-таблице."""
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.Q[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env, do_rendering=False, update_tables=True):
        """Один эпизод жадной стратегии. Возвращает (исходное вознаграждение, длину)."""
        total_reward = 0.0
        steps = 0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if do_rendering:
                print(env.render())
            if update_tables:
                self._record(state, action, new_state, reward, terminated)
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward, steps

    def Q_iteration(self):
        """Одна итерация обновления (2) по всем состояниям и действиям."""
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transition_table[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.reward_table[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (
                        reward + self.gamma * self.Q[(tgt_state, best_action)]
                    )
                self.Q[(state, action)] = action_value

    # ---- вспомогательные методы для анализа ----
    def q_matrix(self):
        n_s, n_a = self.env.observation_space.n, self.env.action_space.n
        return np.array([[self.Q[(s, a)] for a in range(n_a)] for s in range(n_s)])

    def policy(self):
        return np.array([self.select_action(s) for s in range(self.env.observation_space.n)])


def train(gamma=0.9, reward_fn=default_reward, seed=0, random_steps=100,
          max_iters=300, stop_at=0.8, early_stop=True):
    """Цикл обучения из методички. Возвращает агента, кривую обучения и номер
    итерации, на которой средняя награда на тесте впервые превысила stop_at."""
    test_env = gym.make(ENV_NAME)
    test_env.reset(seed=seed + 10_000)
    agent = Agent(gamma=gamma, reward_fn=reward_fn, seed=seed)
    curve, solved_at = [], None
    for iter_no in range(1, max_iters + 1):
        agent.play_n_random_steps(random_steps)
        agent.Q_iteration()
        reward = sum(agent.play_episode(test_env)[0] for _ in range(TEST_EPISODES))
        reward /= TEST_EPISODES
        curve.append(reward)
        if solved_at is None and reward > stop_at:
            solved_at = iter_no
            if early_stop:
                break
    return agent, np.array(curve), solved_at


def evaluate(agent, episodes=10_000, seed=123):
    """Честная оценка стратегии: доля успешных эпизодов, средняя длина эпизода
    и доля эпизодов, закончившихся в полынье."""
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    wins, lengths, holes = 0, [], 0
    for _ in range(episodes):
        state, _ = env.reset()
        steps = 0
        while True:
            state, r, term, trunc, _ = env.step(agent.select_action(state))
            steps += 1
            if term or trunc:
                break
        wins += r
        holes += int(term and r == 0)
        lengths.append(steps)
    return wins / episodes, float(np.mean(lengths)), holes / episodes


def exact_value_iteration(gamma, reward_fn=default_reward, tol=1e-10):
    """Q-итерация по точной модели среды (env.unwrapped.P) — эталон для сравнения."""
    env = gym.make(ENV_NAME).unwrapped
    desc = env.desc.astype(str)
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_s, n_a))
    while True:
        Q_new = np.zeros_like(Q)
        for s in range(n_s):
            for a in range(n_a):
                for p, s2, r, term in env.P[s][a]:
                    if desc.flat[s] in "HG":  # терминальные состояния
                        continue
                    rr = reward_fn(s, a, s2, r, term, desc)
                    Q_new[s, a] += p * (rr + gamma * (0 if term else Q[s2].max()))
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new


def format_policy(policy, desc):
    rows = []
    for i, row in enumerate(desc):
        cells = []
        for j, c in enumerate(row):
            s = i * len(row) + j
            cells.append(c if c in "HG" else ACTION_ARROWS[policy[s]])
        rows.append(" ".join(cells))
    return "\n".join(rows)


if __name__ == "__main__":
    agent, curve, solved = train(gamma=0.9, seed=0)
    print(f"Solved in {solved} iterations!")
    env = gym.make(ENV_NAME, render_mode="ansi")
    env.reset(seed=1)
    agent.play_episode(env, do_rendering=True)
    print(format_policy(agent.policy(), agent.desc))
