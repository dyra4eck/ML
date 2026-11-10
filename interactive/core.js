// ===== Общие утилиты =====
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

// ===== Лаба 1: FrozenLake + итерация по ценностям действий =====
const FL = (() => {
  const MAP = "SFFFFHFHFFFHHFFG", N = 16, A = 4;
  const isHole = s => MAP[s] === "H", isGoal = s => MAP[s] === "G", isTerm = s => isHole(s) || isGoal(s);
  function move(s, a) { // 0 влево, 1 вниз, 2 вправо, 3 вверх
    let r = s >> 2, c = s & 3;
    if (a === 0) c = Math.max(0, c - 1); else if (a === 1) r = Math.min(3, r + 1);
    else if (a === 2) c = Math.min(3, c + 1); else r = Math.max(0, r - 1);
    return r * 4 + c;
  }
  // истинная модель: по 1/3 на выбранное и два перпендикулярных направления
  const outcomes = (s, a) => [(a + 3) % 4, a, (a + 1) % 4].map(d => move(s, d));
  const REWARDS = {
    default: { label: "исходная: +1 за цель", hole: 0, step: 0 },
    hole: { label: "−1 за полынью", hole: -1, step: 0 },
    step001: { label: "−0.01 за шаг", hole: 0, step: -0.01 },
    step02: { label: "−0.2 за шаг", hole: 0, step: -0.2 },
    both: { label: "−1 за полынью и −0.01 за шаг", hole: -1, step: -0.01 },
  };
  function shaped(kind, s2) {
    const R = REWARDS[kind];
    if (isGoal(s2)) return 1;
    if (isHole(s2) && R.hole) return R.hole;
    return R.step;
  }
  function argmax(q) { let b = 0; for (let a = 1; a < A; a++) if (q[a] > q[b]) b = a; return b; }

  class Agent {
    constructor(gamma, kind, seed) {
      this.gamma = gamma; this.kind = kind; this.rnd = mulberry32(seed);
      this.counts = Array.from({ length: N }, () => Array.from({ length: A }, () => new Array(N).fill(0)));
      this.Q = Array.from({ length: N }, () => new Array(A).fill(0));
      this.state = 0; this.t = 0; this.samples = 0; this.iter = 0; this.history = [];
    }
    envStep(s, a) { return outcomes(s, a)[Math.floor(this.rnd() * 3)]; }
    record(s, a, s2) { this.counts[s][a][s2]++; }
    collect(n) { // n случайных действий
      const fresh = [];
      for (let i = 0; i < n; i++) {
        const a = Math.floor(this.rnd() * A), s2 = this.envStep(this.state, a);
        this.record(this.state, a, s2); fresh.push([this.state, a, s2]); this.samples++; this.t++;
        if (isTerm(s2) || this.t >= 100) { this.state = 0; this.t = 0; } else this.state = s2;
      }
      return fresh;
    }
    qValue(s, a) { // правая часть уравнения Беллмана по накопленной статистике
      const c = this.counts[s][a], total = c.reduce((x, y) => x + y, 0), terms = [];
      let v = 0;
      if (total) for (let s2 = 0; s2 < N; s2++) if (c[s2]) {
        const p = c[s2] / total, r = shaped(this.kind, s2), next = Math.max(...this.Q[s2]);
        terms.push({ s2, n: c[s2], total, p, r, next }); v += p * (r + this.gamma * next);
      }
      return { v, terms, total };
    }
    qIteration() { // один проход по всем (s, a), как в Q_iteration()
      let delta = 0;
      for (let s = 0; s < N; s++) for (let a = 0; a < A; a++) {
        const v = this.qValue(s, a).v; delta = Math.max(delta, Math.abs(v - this.Q[s][a])); this.Q[s][a] = v;
      }
      return delta;
    }
    policy() { return this.Q.map(argmax); }
    episode(learn = true) { // жадный эпизод; возвращает путь
      let s = 0; const path = [];
      for (let t = 0; t < 100; t++) {
        const a = argmax(this.Q[s]), s2 = this.envStep(s, a);
        if (learn) this.record(s, a, s2);
        path.push({ s, a, s2 }); s = s2; if (isTerm(s)) break;
      }
      return { path, win: isGoal(s), hole: isHole(s) };
    }
    test(n = 20) { let w = 0; for (let i = 0; i < n; i++) w += this.episode().win ? 1 : 0; return w; }
    iteration() {
      this.collect(100); this.qIteration(); const wins = this.test(20); this.iter++;
      const st = policyStats(this.policy());
      this.history.push({ iter: this.iter, test: wins / 20, pGoal: st.pGoal });
      return wins;
    }
  }
  function policyStats(pol) { // точный расчёт по истинной модели, 100 шагов
    let dist = new Array(N).fill(0); dist[0] = 1; let pGoal = 0, pHole = 0, len = 0;
    for (let t = 0; t < 100; t++) {
      const nd = new Array(N).fill(0); let alive = 0;
      for (let s = 0; s < N; s++) if (dist[s]) { alive += dist[s]; for (const s2 of outcomes(s, pol[s])) nd[s2] += dist[s] / 3; }
      len += alive;
      for (let s = 0; s < N; s++) { if (isGoal(s)) { pGoal += nd[s]; nd[s] = 0; } else if (isHole(s)) { pHole += nd[s]; nd[s] = 0; } }
      dist = nd;
    }
    return { pGoal, pHole, len };
  }
  function optimal(gamma, kind) { // итерация по точной модели — эталон
    let Q = Array.from({ length: N }, () => new Array(A).fill(0));
    for (let it = 0; it < 5000; it++) {
      let d = 0; const Q2 = Q.map(r => r.slice());
      for (let s = 0; s < N; s++) if (!isTerm(s)) for (let a = 0; a < A; a++) {
        let v = 0; for (const s2 of outcomes(s, a)) v += (shaped(kind, s2) + gamma * (isTerm(s2) ? 0 : Math.max(...Q[s2]))) / 3;
        d = Math.max(d, Math.abs(v - Q[s][a])); Q2[s][a] = v;
      }
      Q = Q2; if (d < 1e-10) break;
    }
    return policyStats(Q.map(argmax));
  }
  return { MAP, N, A, isHole, isGoal, isTerm, outcomes, REWARDS, shaped, argmax, Agent, policyStats, optimal };
})();

// ===== Лаба 2: DQN на упрощённом (вертикальном) посадочном модуле =====
const DQN = (() => {
  // --- среда: высота y, скорость v; действия: 0 — ничего, 1 — главный двигатель ---
  const G = -0.004, THRUST = 0.010, SAFE_V = 0.03, MAX_T = 200, TOP = 1.6;
  const obs = e => [e.y, e.v * 10];
  const phi = e => -100 * e.y - 400 * Math.abs(e.v);
  function reset(rnd) { return { y: 0.9 + 0.3 * rnd(), v: -0.04 * rnd(), t: 0 }; }
  function step(e, a) {
    const before = phi(e);
    const n = { v: e.v + G + (a ? THRUST : 0), t: e.t + 1 }; n.y = e.y + n.v;
    let r, done = false, outcome = null;
    if (n.y <= 0) { n.y = 0; done = true; outcome = Math.abs(n.v) <= SAFE_V ? "landed" : "crashed"; r = -100 * 0 - before - 400 * Math.abs(n.v) + (outcome === "landed" ? 100 : -100); }
    else if (n.y > TOP) { done = true; outcome = "flew"; r = phi(n) - before - 100; }
    else r = phi(n) - before;
    if (a) r -= 0.3;
    const truncated = !done && n.t >= MAX_T;
    if (truncated) outcome = "timeout";
    return { env: n, r, done, truncated, outcome };
  }
  // --- маленькая полносвязная сеть 2 → H → H → 2 с Adam ---
  class Net {
    constructor(h, rnd) {
      const init = (n, m) => Float64Array.from({ length: n * m }, () => (rnd() * 2 - 1) * Math.sqrt(6 / m));
      this.h = h; this.W1 = init(h, 2); this.b1 = new Float64Array(h); this.W2 = init(h, h); this.b2 = new Float64Array(h);
      this.W3 = init(2, h); this.b3 = new Float64Array(2);
      this.params = [this.W1, this.b1, this.W2, this.b2, this.W3, this.b3];
      this.m = this.params.map(p => new Float64Array(p.length)); this.vv = this.params.map(p => new Float64Array(p.length)); this.step = 0;
    }
    forward(x, keep) {
      const h = this.h, a1 = new Float64Array(h), a2 = new Float64Array(h), q = new Float64Array(2);
      for (let i = 0; i < h; i++) { const z = this.W1[i * 2] * x[0] + this.W1[i * 2 + 1] * x[1] + this.b1[i]; a1[i] = z > 0 ? z : 0; }
      for (let i = 0; i < h; i++) { let z = this.b2[i]; for (let j = 0; j < h; j++) z += this.W2[i * h + j] * a1[j]; a2[i] = z > 0 ? z : 0; }
      for (let k = 0; k < 2; k++) { let z = this.b3[k]; for (let j = 0; j < h; j++) z += this.W3[k * h + j] * a2[j]; q[k] = z; }
      if (keep) return { x, a1, a2, q };
      return q;
    }
    copyFrom(o) { this.params.forEach((p, i) => p.set(o.params[i])); }
    // один шаг градиентного спуска по пакету: loss = mean (Q(s,a) − target)²
    train(batch, lr) {
      const h = this.h, g = this.params.map(p => new Float64Array(p.length)); let loss = 0;
      for (const { x, a, target } of batch) {
        const f = this.forward(x, true), err = f.q[a] - target; loss += err * err;
        const dq = 2 * err / batch.length, d2 = new Float64Array(h), d1 = new Float64Array(h);
        g[5][a] += dq;
        for (let j = 0; j < h; j++) { g[4][a * h + j] += dq * f.a2[j]; if (f.a2[j] > 0) d2[j] = dq * this.W3[a * h + j]; }
        for (let i = 0; i < h; i++) if (d2[i]) { g[3][i] += d2[i]; for (let j = 0; j < h; j++) { g[2][i * h + j] += d2[i] * f.a1[j]; if (f.a1[j] > 0) d1[j] += d2[i] * this.W2[i * h + j]; } }
        for (let i = 0; i < h; i++) if (d1[i]) { g[1][i] += d1[i]; g[0][i * 2] += d1[i] * x[0]; g[0][i * 2 + 1] += d1[i] * x[1]; }
      }
      this.step++; const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, this.step), c2 = 1 - Math.pow(b2, this.step);
      this.params.forEach((p, k) => { const m = this.m[k], v = this.vv[k], gk = g[k];
        for (let i = 0; i < p.length; i++) { const gi = Math.max(-10, Math.min(10, gk[i])); m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi; p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); } });
      return loss / batch.length;
    }
  }
  const DEFAULTS = { gamma: 0.99, lr: 1e-3, batch: 32, buffer: 20000, sync: 200, epsDecay: 0.97, epsFinal: 0.02, epsStart: 1, hidden: 32, useTarget: true, useReplay: true };
  class Agent {
    constructor(cfg, seed) {
      this.cfg = Object.assign({}, DEFAULTS, cfg); this.rnd = mulberry32(seed);
      this.Q = new Net(this.cfg.hidden, this.rnd); this.T = new Net(this.cfg.hidden, this.rnd); this.T.copyFrom(this.Q);
      this.buf = []; this.ptr = 0; this.eps = this.cfg.epsStart; this.trainSteps = 0; this.syncs = 0; this.episodes = 0;
      this.history = []; this.lastLoss = 0; this.env = reset(this.rnd); this.epReward = 0; this.epLoss = 0; this.epSteps = 0;
    }
    get capacity() { return this.cfg.useReplay ? this.cfg.buffer : this.cfg.batch; }
    act(x, greedy) { if (!greedy && this.rnd() < this.eps) return { a: this.rnd() < 0.5 ? 0 : 1, random: true }; const q = this.Q.forward(x); return { a: q[1] > q[0] ? 1 : 0, random: false, q }; }
    push(tr) { if (this.buf.length < this.capacity) this.buf.push(tr); else { this.ptr %= this.capacity; this.buf[this.ptr] = tr; } this.ptr = (this.ptr + 1) % this.capacity; }
    explain(tr) { // разбор одного перехода: предсказание, цель, ошибка
      const pred = this.Q.forward(tr.x)[tr.a], qn = this.T.forward(tr.x2), nextMax = tr.done ? 0 : Math.max(qn[0], qn[1]);
      const target = tr.r + this.cfg.gamma * nextMax; return { pred, qn, nextMax, target, err: target - pred };
    }
    trainStep() {
      const c = this.cfg; if (this.buf.length < c.batch) return null;
      const batch = [];
      for (let i = 0; i < c.batch; i++) { const tr = this.buf[Math.floor(this.rnd() * this.buf.length)]; batch.push({ x: tr.x, a: tr.a, target: this.explain(tr).target }); }
      this.lastLoss = this.Q.train(batch, c.lr); this.trainSteps++;
      if (!c.useTarget || this.trainSteps % c.sync === 0) { this.T.copyFrom(this.Q); this.syncs++; }
      return this.lastLoss;
    }
    envStep() { // один шаг взаимодействия + один шаг обучения
      const x = obs(this.env), { a, random } = this.act(x), res = step(this.env, a);
      const tr = { x, a, r: res.r, x2: obs(res.env), done: res.done };
      this.push(tr); this.env = res.env; this.epReward += res.r; this.epSteps++;
      const loss = this.trainStep(); if (loss != null) this.epLoss += loss;
      const out = { tr, random, res, finished: res.done || res.truncated };
      if (out.finished) {
        this.episodes++; this.history.push({ ep: this.episodes, reward: this.epReward, outcome: res.outcome, eps: this.eps, loss: this.epLoss / this.epSteps });
        this.eps = Math.max(this.cfg.epsFinal, this.eps * this.cfg.epsDecay);
        this.env = reset(this.rnd); this.epReward = 0; this.epLoss = 0; this.epSteps = 0;
      }
      return out;
    }
    runEpisode() { let o; do { o = this.envStep(); } while (!o.finished); return this.history[this.history.length - 1]; }
    greedyEpisode(seed) { const rnd = mulberry32(seed); let e = reset(rnd), total = 0; const frames = [];
      for (;;) { const { a, q } = this.act(obs(e), true), res = step(e, a); frames.push({ y: e.y, v: e.v, a, q }); total += res.r; e = res.env; if (res.done || res.truncated) { frames.push({ y: e.y, v: e.v, a: 0 }); return { frames, total, outcome: res.outcome }; } } }
    evaluate(n = 30) { let s = 0, ok = 0; for (let i = 0; i < n; i++) { const r = this.greedyEpisode(1000 + i); s += r.total; ok += r.outcome === "landed" ? 1 : 0; } return { mean: s / n, landed: ok / n }; }
  }
  return { reset, step, obs, Net, Agent, DEFAULTS, SAFE_V, TOP, MAX_T };
})();
if (typeof module !== "undefined") module.exports = { FL, DQN, mulberry32 };
