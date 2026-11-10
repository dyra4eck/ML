const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (x, d = 2) => (Math.abs(x) < 0.0005 ? 0 : x).toFixed(d).replace("-", "−");
const ARROWS = ["←", "↓", "→", "↑"], ANAMES = ["влево", "вниз", "вправо", "вверх"];

// ---------- вкладки ----------
function showTab(name) {
  for (const t of ["lab1", "lab2"]) { $(t).hidden = t !== name; $("tab-" + t).setAttribute("aria-selected", t === name); }
  if (name === "lab2") L2.draw();
}
document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => { location.hash = b.dataset.tab; showTab(b.dataset.tab); }));
window.addEventListener("hashchange", () => showTab(location.hash === "#lab2" ? "lab2" : "lab1"));

// ---------- линейный график (SVG) с подсказкой ----------
function lineChart(box, { series, yMin, yMax, ref, xLabel, tip }) {
  const W = 600, H = 230, L = 40, R = 12, T = 10, B = 30;
  const n = Math.max(1, ...series.map(s => s.pts.length)), xMax = Math.max(10, n);
  const X = i => L + (W - L - R) * (i / xMax), Y = v => T + (H - T - B) * (1 - (Math.min(yMax, Math.max(yMin, v)) - yMin) / (yMax - yMin));
  let g = "";
  for (let k = 0; k <= 4; k++) { const v = yMin + (yMax - yMin) * k / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="1"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="var(--ink-2)">${+v.toFixed(2)}</text>`; }
  for (let k = 0; k <= 5; k++) { const i = Math.round(xMax * k / 5); g += `<text x="${X(i)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--ink-2)">${i}</text>`; }
  g += `<text x="${W - R}" y="${H - 1}" text-anchor="end" font-size="11" fill="var(--ink-2)">${xLabel}</text>`;
  if (ref != null) g += `<line x1="${L}" x2="${W - R}" y1="${Y(ref)}" y2="${Y(ref)}" stroke="var(--ink-2)" stroke-width="1" stroke-dasharray="4 4"/>`;
  for (const s of series) if (s.pts.length) {
    g += `<polyline fill="none" stroke="${s.color}" stroke-width="${s.w}" stroke-linejoin="round" points="${s.pts.map((v, i) => X(i + 1) + "," + Y(v)).join(" ")}"/>`;
    if (s.w >= 2) { const i = s.pts.length; g += `<circle cx="${X(i)}" cy="${Y(s.pts[i - 1])}" r="4" fill="${s.color}" stroke="var(--panel)" stroke-width="2"/>`; }
  }
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}<line id="x" y1="${T}" y2="${H - B}" stroke="var(--ink-2)" stroke-width="1" visibility="hidden"/></svg><div class="tip" hidden></div>`;
  const svg = box.firstChild, tipEl = box.lastChild, cross = svg.querySelector("#x");
  svg.onmousemove = e => { const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
    const i = Math.round((px - L) / (W - L - R) * xMax); if (i < 1 || i > n || !series[0].pts.length) { tipEl.hidden = true; cross.setAttribute("visibility", "hidden"); return; }
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    tipEl.hidden = false; tipEl.textContent = tip(i - 1); tipEl.style.left = (X(i) / W * 100) + "%"; tipEl.style.top = "30%"; };
  svg.onmouseleave = () => { tipEl.hidden = true; cross.setAttribute("visibility", "hidden"); };
}

// ================= ЛАБА 1 =================
const L1 = (() => {
  let ag, sel = 0, walker = null, playing = false, seed = 1, opt;
  const rsel = $("l1-reward");
  for (const k in FL.REWARDS) rsel.add(new Option(FL.REWARDS[k].label, k));
  function reset() { ag = new FL.Agent(+$("l1-gamma").value, rsel.value, seed++); opt = FL.optimal(ag.gamma, ag.kind); walker = null;
    log("Все Q равны нулю: агент ничего не знает. Начни с шага 1."); draw(); }
  const log = t => $("l1-log").innerHTML = t;
  function draw() {
    const V = ag.Q.map(q => Math.max(...q)), pol = ag.policy(), view = $("l1-view").value;
    const visits = ag.counts.map(c => c.reduce((s, a) => s + a.reduce((x, y) => x + y, 0), 0));
    const vmax = Math.max(0.001, ...V.map(Math.abs)), nmax = Math.max(1, ...visits);
    let g = "";
    for (let s = 0; s < 16; s++) {
      const x = (s & 3) * 100, y = (s >> 2) * 100, hole = FL.isHole(s), goal = FL.isGoal(s);
      let fill = "var(--panel)", label = "", sub = "";
      if (hole) { fill = "color-mix(in srgb, var(--bad) 30%, var(--panel))"; label = "H"; sub = "полынья"; }
      else if (goal) { fill = "color-mix(in srgb, var(--good) 30%, var(--panel))"; label = "G"; sub = "цель"; }
      else if (view === "v") { const k = Math.round(55 * Math.abs(V[s]) / vmax); fill = `color-mix(in srgb, var(--${V[s] < 0 ? "bad" : "accent"}) ${k}%, var(--panel))`;
        label = V[s] === 0 && visits[s] === 0 ? "·" : ARROWS[pol[s]]; sub = fmt(V[s], Math.abs(V[s]) < 0.1 ? 3 : 2); }
      else { fill = `color-mix(in srgb, var(--accent) ${Math.round(55 * visits[s] / nmax)}%, var(--panel))`; label = visits[s]; sub = "шагов отсюда"; }
      if (view === "n" && (hole || goal)) sub = hole ? "полынья" : "цель";
      g += `<g class="cell" tabindex="0" data-s="${s}" role="button" aria-label="клетка ${s}">
        <rect x="${x + 2}" y="${y + 2}" width="96" height="96" rx="8" fill="${fill}" stroke="${s === sel ? "var(--ink)" : "var(--line)"}" stroke-width="${s === sel ? 3 : 1}"/>
        <text x="${x + 10}" y="${y + 20}" font-size="11" style="fill:var(--ink-2)">${s}${s === 0 ? " старт" : ""}</text>
        <text x="${x + 50}" y="${y + 60}" text-anchor="middle" font-size="${view === "n" && !hole && !goal ? 22 : 30}" font-weight="600">${label}</text>
        <text x="${x + 50}" y="${y + 84}" text-anchor="middle" font-size="12">${sub}</text></g>`;
    }
    if (walker != null) g += `<circle cx="${(walker & 3) * 100 + 78}" cy="${(walker >> 2) * 100 + 24}" r="11" fill="var(--warn)" stroke="var(--panel)" stroke-width="3"/>`;
    $("l1-grid").innerHTML = `<svg viewBox="0 0 400 400">${g}</svg>`;
    const st = FL.policyStats(pol), last = ag.history[ag.history.length - 1];
    $("l1-stats").innerHTML = [[ag.iter, "итераций"], [ag.samples, "случайных шагов"], [last ? Math.round(last.test * 20) + "/20" : "—", "последний тест"],
      [fmt(st.pGoal), "дойдёт до цели (точно)"], [fmt(st.pHole), "упадёт в полынью"], [fmt(st.len, 1), "шагов в эпизоде"], [fmt(opt.pGoal), "лучшее возможное при этих γ и награде"]]
      .map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    breakdown();
    lineChart($("l1-chart"), { series: [{ color: "var(--raw)", w: 1, pts: ag.history.map(h => h.test) }, { color: "var(--accent)", w: 2, pts: ag.history.map(h => h.pGoal) }],
      yMin: 0, yMax: 1, ref: 0.8, xLabel: "итерация", tip: i => `итерация ${i + 1}: тест ${ag.history[i].test.toFixed(2)}, точно ${ag.history[i].pGoal.toFixed(2)}` });
  }
  function breakdown() {
    $("l1-bd-title").textContent = `Разбор клетки ${sel}` + (sel === 0 ? " (старт)" : "");
    if (FL.isTerm(sel)) { $("l1-bd").innerHTML = `<div class="act">Это ${FL.isGoal(sel) ? "цель" : "полынья"}: эпизод здесь заканчивается, действий нет, поэтому её ценность всегда 0. Награду агент получает за <i>переход</i> в эту клетку.</div>`; return; }
    const best = FL.argmax(ag.Q[sel]);
    $("l1-bd").innerHTML = [0, 1, 2, 3].map(a => { const r = ag.qValue(sel, a);
      const terms = r.terms.length ? r.terms.map(t => `<div class="t">${t.n} из ${t.total} раз → клетка ${t.s2}${FL.isHole(t.s2) ? " (H)" : FL.isGoal(t.s2) ? " (G)" : ""}:  ${fmt(t.p)} × (${fmt(t.r)} + ${ag.gamma} × ${fmt(t.next, 3)})</div>`).join("")
        : `<div class="t">данных нет: это действие отсюда ещё не пробовали</div>`;
      const changed = Math.abs(r.v - ag.Q[sel][a]) > 5e-4;
      return `<div class="act mono ${a === best && ag.samples ? "best" : ""}"><b>${ARROWS[a]} ${ANAMES[a]}: Q = ${fmt(ag.Q[sel][a], 3)}</b>${a === best && ag.samples ? " · лучшее" : ""}${changed ? ` · после пересчёта станет ${fmt(r.v, 3)}` : ""}${terms}</div>`; }).join("");
  }
  function doTest() { const w = ag.test(20); ag.iter++; const st = FL.policyStats(ag.policy()); ag.history.push({ iter: ag.iter, test: w / 20, pGoal: st.pGoal }); return w; }
  $("l1-collect").onclick = () => { const f = ag.collect(100), goals = f.filter(x => FL.isGoal(x[2])).length, holes = f.filter(x => FL.isHole(x[2])).length;
    log(`Сделано 100 случайных шагов (всего ${ag.samples}). В цель попал ${goals} раз, в полынью ${holes}. Q пока не изменились — нажми шаг 2.`); draw(); };
  $("l1-q").onclick = () => { const d = ag.qIteration(); log(d < 1e-9 ? "Пересчёт ничего не изменил. Нужны новые данные (шаг 1) — или агент ещё ни разу не видел цель." : `Q пересчитаны. Самое большое изменение: ${fmt(d, 3)}. Ценность распространяется от цели к соседним клеткам.`); draw(); };
  $("l1-test").onclick = () => { const w = doTest(); log(`Тест: ${w} из 20 эпизодов дошли до цели (reward = ${(w / 20).toFixed(2)}). Точка добавлена на график.`); draw(); };
  const iter = n => { let w; for (let i = 0; i < n; i++) w = ag.iteration(); log(`Сделано итераций: ${n}. Последний тест: ${w} из 20.`); draw(); };
  $("l1-iter").onclick = () => iter(1); $("l1-iter10").onclick = () => iter(10);
  $("l1-play").onclick = () => { if (playing) return; playing = true; const ep = ag.episode(false); let i = 0; walker = 0; draw();
    const tick = () => { if (i >= ep.path.length || i >= 40) { playing = false; log(ep.win ? `Дошёл до цели за ${ep.path.length} шагов.` : ep.hole ? `Провалился в полынью на шаге ${ep.path.length}.` : i >= 40 && !FL.isTerm(walker) ? "Показаны первые 40 шагов, агент всё ещё идёт." : "Не успел за 100 шагов."); walker = null; setTimeout(draw, 900); return; }
      const p = ep.path[i++]; walker = p.s2; const want = FL.outcomes(p.s, p.a)[1];
      log(`Шаг ${i}: в клетке ${p.s} выбрал ${ARROWS[p.a]} ${ANAMES[p.a]} — ${p.s2 === want ? "пошёл как хотел" : "поскользнулся"}, оказался в клетке ${p.s2}.`); draw(); setTimeout(tick, 450); };
    setTimeout(tick, 450); };
  $("l1-reset").onclick = reset; $("l1-gamma").onchange = reset; rsel.onchange = reset; $("l1-view").onchange = draw;
  $("l1-grid").addEventListener("click", e => { const c = e.target.closest(".cell"); if (c) { sel = +c.dataset.s; draw(); } });
  $("l1-grid").addEventListener("keydown", e => { const c = e.target.closest(".cell"); if (c && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sel = +c.dataset.s; draw(); } });
  reset();
  return { draw };
})();

// ================= ЛАБА 2 =================
const L2 = (() => {
  let ag, seed = 1, auto = false, busy = false, view, loupe = null, outcome = null;
  const OUT = { landed: ["мягкая посадка", "good"], crashed: ["разбился", "bad"], flew: ["улетел вверх", "bad"], timeout: ["завис до конца времени", "warn"] };
  const log = t => $("l2-log").innerHTML = t;
  function reset() { const ex = $("l2-explore").value;
    ag = new DQN.Agent({ useReplay: $("l2-replay").checked, useTarget: $("l2-target").checked, epsDecay: ex === "none" ? 1 : +ex, epsStart: ex === "none" ? 0.02 : 1 }, seed++);
    view = { y: ag.env.y, v: ag.env.v, a: 0 }; loupe = null; outcome = null; auto = false; $("l2-auto").textContent = "Авто";
    log("Сеть случайная. Нажми «1 эпизод» и посмотри, что делает необученный агент."); draw(); }
  function drawLander() {
    const c = $("l2-lander"), x = c.getContext("2d"), W = c.width, H = c.height, gy = H - 34, py = y => gy - y / 1.6 * (gy - 20);
    x.clearRect(0, 0, W, H); x.fillStyle = css("--line"); x.fillRect(0, gy, W, H - gy);
    x.fillStyle = css("--good"); x.fillRect(W / 2 - 60, gy, 120, 5);
    x.strokeStyle = css("--line"); x.setLineDash([4, 4]); x.beginPath(); x.moveTo(0, py(1.6)); x.lineTo(W, py(1.6)); x.stroke(); x.setLineDash([]);
    x.fillStyle = css("--ink-2"); x.font = "12px Inter, sans-serif"; x.fillText("выше — вылет за границу", 8, py(1.6) + 14); x.fillText("площадка", W / 2 + 68, gy + 16);
    const cy = py(view.y) - 14, cx = W / 2;
    if (view.a) { x.fillStyle = css("--warn"); x.beginPath(); x.moveTo(cx - 8, cy + 14); x.lineTo(cx + 8, cy + 14); x.lineTo(cx, cy + 36); x.fill(); }
    x.fillStyle = css("--accent"); x.beginPath(); x.moveTo(cx - 14, cy + 14); x.lineTo(cx - 10, cy - 10); x.lineTo(cx + 10, cy - 10); x.lineTo(cx + 14, cy + 14); x.fill();
    x.strokeStyle = css("--ink"); x.lineWidth = 2; x.beginPath(); x.moveTo(cx - 12, cy + 12); x.lineTo(cx - 20, cy + 14); x.moveTo(cx + 12, cy + 12); x.lineTo(cx + 20, cy + 14); x.stroke();
    // стрелка скорости
    const len = Math.max(-60, Math.min(60, -view.v * 600)); x.strokeStyle = css("--ink-2"); x.beginPath(); x.moveTo(cx + 44, cy); x.lineTo(cx + 44, cy + len); x.stroke();
    x.fillStyle = css("--ink-2"); x.fillText("скорость", cx + 52, cy + 4);
  }
  const VMIN = -0.12, VMAX = 0.06, YMAX = 1.5;
  function drawMap() {
    const c = $("l2-map"), x = c.getContext("2d"), W = c.width, H = c.height, nx = 48, ny = 30, on = css("--accent"), off = css("--line");
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const v = VMIN + (VMAX - VMIN) * (i + .5) / nx, y = YMAX * (1 - (j + .5) / ny), q = ag.Q.forward([y, v * 10]);
      x.fillStyle = q[1] > q[0] ? on : off; x.fillRect(Math.floor(i * W / nx), Math.floor(j * H / ny), Math.ceil(W / nx), Math.ceil(H / ny)); }
    const zx = (0 - VMIN) / (VMAX - VMIN) * W; x.strokeStyle = css("--ink-2"); x.lineWidth = 1; x.setLineDash([3, 3]); x.beginPath(); x.moveTo(zx, 0); x.lineTo(zx, H); x.stroke(); x.setLineDash([]);
    x.fillStyle = css("--ink"); x.font = "12px Inter, sans-serif"; x.fillText("скорость 0", zx + 6, 14); x.fillText("земля", 6, H - 6); x.fillText("высоко", 6, 14);
    const px = (Math.max(VMIN, Math.min(VMAX, view.v)) - VMIN) / (VMAX - VMIN) * W, py = (1 - Math.min(YMAX, view.y) / YMAX) * H;
    x.fillStyle = css("--warn"); x.strokeStyle = css("--panel"); x.lineWidth = 3; x.beginPath(); x.arc(px, py, 7, 0, 7); x.fill(); x.stroke();
  }
  function drawLoupe() {
    if (!loupe) return; const t = loupe, e = ag.explain(t), g = ag.cfg.gamma;
    const s = x => `высота ${fmt(x[0])}, скорость ${fmt(x[1] / 10, 3)}`;
    $("l2-loupe").innerHTML = `<div class="act"><div class="t">состояние s:   ${s(t.x)}</div><div class="t">действие a:    ${t.a ? "двигатель" : "ничего"}</div><div class="t">награда r:     ${fmt(t.r)}</div><div class="t">дальше s′:     ${s(t.x2)}${t.done ? " — КОНЕЦ эпизода" : ""}</div></div>
      <div class="act"><div class="t">сеть предсказала  Q(s, a) = <b>${fmt(e.pred)}</b></div>
      <div class="t">цель = r + γ × max Q⁻(s′) = ${fmt(t.r)} + ${t.done ? "0 (будущего нет)" : `${g} × ${fmt(e.nextMax)}`} = <b>${fmt(e.target)}</b></div>
      <div class="t">ошибка = цель − предсказание = <b>${fmt(e.err)}</b></div></div>
      <div class="act" style="white-space:normal;font-family:var(--font-body)">${Math.abs(e.err) < 1 ? "Предсказание почти совпало с целью: тут сеть уже «согласована» с уравнением Беллмана." : `Сеть ${e.err > 0 ? "недооценила" : "переоценила"} это действие. Шаг обучения сдвинет Q(s, a) ${e.err > 0 ? "вверх" : "вниз"}.`}</div>`;
  }
  function draw() {
    if ($("lab2").hidden) return; drawLander(); drawMap(); drawLoupe();
    const h = ag.history, l = h.slice(-20), avg = l.length ? l.reduce((s, e) => s + e.reward, 0) / l.length : null;
    $("l2-stats").innerHTML = [[ag.episodes, "эпизодов"], [avg == null ? "—" : fmt(avg, 0), "средняя награда за 20"], [fmt(ag.eps), "ε сейчас"],
      [ag.buf.length, "переходов в буфере"], [ag.trainSteps, "шагов обучения"], [ag.syncs, "копирований в целевую сеть"]].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    $("l2-now").innerHTML = [[fmt(view.y), "высота"], [fmt(view.v, 3), "скорость (мягко: до ±0.03)"], [view.a ? "вкл" : "выкл", "двигатель"]].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    $("l2-epsbar").style.width = ag.eps * 100 + "%"; $("l2-bufbar").style.width = Math.min(100, ag.buf.length / Math.min(ag.capacity, 5000) * 100) + "%";
    const o = outcome && OUT[outcome]; $("l2-outcome").className = "pill " + (o ? o[1] : ""); $("l2-outcome").textContent = o ? o[0] : "в полёте";
    const avgs = h.map((_, i) => { const w = h.slice(Math.max(0, i - 19), i + 1); return w.reduce((s, e) => s + e.reward, 0) / w.length; });
    lineChart($("l2-chart"), { series: [{ color: "var(--raw)", w: 1, pts: h.map(e => e.reward) }, { color: "var(--accent)", w: 2, pts: avgs }], yMin: -250, yMax: 250, ref: 150, xLabel: "эпизод",
      tip: i => `эпизод ${i + 1}: ${h[i].reward.toFixed(0)} (${OUT[h[i].outcome][0]}), среднее ${avgs[i].toFixed(0)}` });
  }
  function one() { const before = ag.env, o = ag.envStep(); loupe = o.tr; view = { y: o.res.env.y, v: o.res.env.v, a: o.tr.a }; outcome = o.finished ? o.res.outcome : null; return { o, before }; }
  function describe(o) { const h = ag.history[ag.history.length - 1]; return `Эпизод ${h.ep}: <b>${OUT[h.outcome][0]}</b>, награда ${fmt(h.reward, 0)}. Случайных действий было около ${Math.round(h.eps * 100)}%.`; }
  function lock(v) { busy = v; for (const id of ["l2-step", "l2-ep", "l2-ep10", "l2-watch"]) $(id).disabled = v; }
  $("l2-step").onclick = () => { const { o } = one(); log(o.finished ? describe(o) : `Шаг: действие «${o.tr.a ? "двигатель" : "ничего"}» ${o.random ? "выбрано <b>случайно</b> (ε)" : "выбрала <b>сеть</b>"}, награда ${fmt(o.tr.r)}. Переход записан в буфер${ag.trainSteps ? ", сделан шаг обучения по 32 случайным переходам" : " (обучение начнётся, когда наберётся 32)"}.`); draw(); };
  $("l2-ep").onclick = () => { if (busy) return; lock(true); const tick = () => { const { o } = one(); draw(); if (o.finished) { log(describe(o)); lock(false); } else requestAnimationFrame(tick); }; tick(); };
  function batch(n, done) { let i = 0; const tick = () => { const t = performance.now(); while (i < n && performance.now() - t < 30) { ag.runEpisode(); i++; }
      const h = ag.history[ag.history.length - 1]; outcome = h.outcome; view = { y: ag.env.y, v: ag.env.v, a: 0 }; loupe = ag.buf[Math.floor(Math.random() * ag.buf.length)]; draw();
      if (i < n && (!done || auto)) setTimeout(tick, 0); else done ? done() : (log(describe()), lock(false)); }; tick(); }
  $("l2-ep10").onclick = () => { if (busy) return; lock(true); batch(10); };
  $("l2-auto").onclick = () => { auto = !auto; $("l2-auto").textContent = auto ? "Стоп" : "Авто"; if (!auto) return; lock(true);
    const loop = () => { if (!auto || ag.episodes >= 400) { auto = false; $("l2-auto").textContent = "Авто"; lock(false); log(describe()); return; } batch(1, () => setTimeout(loop, 0)); }; loop(); };
  $("l2-watch").onclick = () => { if (busy) return; lock(true); const ep = ag.greedyEpisode(Math.floor(Math.random() * 1e6)); let i = 0;
    const tick = () => { const f = ep.frames[Math.min(i, ep.frames.length - 1)]; view = { y: f.y, v: f.v, a: f.a }; outcome = i >= ep.frames.length - 1 ? ep.outcome : null; drawLander(); drawMap();
      const o = outcome && OUT[outcome]; $("l2-outcome").className = "pill " + (o ? o[1] : ""); $("l2-outcome").textContent = o ? o[0] : "в полёте";
      if (++i < ep.frames.length) requestAnimationFrame(tick); else { log(`Без случайных действий: <b>${OUT[ep.outcome][0]}</b>, награда ${fmt(ep.total, 0)}. Это то, что сеть умеет на самом деле сейчас.`); lock(false); draw(); } }; tick(); };
  $("l2-sample").onclick = () => { if (ag.buf.length) { loupe = ag.buf[Math.floor(Math.random() * ag.buf.length)]; drawLoupe(); } };
  $("l2-reset").onclick = () => { auto = false; lock(false); reset(); };
  reset();
  return { draw };
})();
showTab(location.hash === "#lab2" ? "lab2" : "lab1");
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => L2.draw());
