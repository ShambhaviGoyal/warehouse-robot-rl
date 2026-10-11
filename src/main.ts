import './style.css';
import './fonts.css';
import './themes.css';
import { ALGO_INFO, Algo, Hyper, TabularAgent } from './agents';
import { ACTIONS, Cell, Layout, WarehouseEnv, encode, mulberry32, sameCell, shelfSet, validateLayout } from './env';
import { BenchmarkRun, EvalResult, Frame, evaluate, mean, movingAverage, rollout, runBenchmark, std } from './evaluate';
import { Plan, optimalPath, solve } from './planner';
import { PRESETS, presetById } from './presets';
import { Series, cssVar, downsample, drawChart } from './charts';
import { RobotPose, ViewState, cellFromEvent, loadSprites, renderGrid } from './view';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

/** Series colours come from the active theme (7-digit hex so alpha can be appended). */
const algoColor = (a: Algo): string => cssVar(`--algo-${a}`, '#38bdf8');
const C = {
  main: () => cssVar('--series', '#38bdf8'),
  good: () => cssVar('--good', '#34d399'),
  warm: () => cssVar('--accent-2', '#fbbf24'),
  muted: () => cssVar('--muted', '#8b97b5'),
};
const SPEEDS = [1, 2, 5, 10, 25, 50, 100, 250];
const ARROWS = ['↑', '↓', '←', '→', 'pick', 'drop'];

const THEMES = ['industrial', 'paper'];
const readStoredTheme = (): string | null => {
  try {
    return localStorage.getItem('rl-lab-theme');
  } catch {
    return null; // storage can be blocked (private windows, embeds)
  }
};
const pickTheme = [new URLSearchParams(location.search).get('theme'), readStoredTheme()].find((t) => THEMES.includes(t ?? ''));
document.documentElement.dataset.theme = pickTheme ?? THEMES[0];

// ---------------------------------------------------------------- state
interface History {
  rewards: number[];
  evalX: number[];
  success: number[];
  efficiency: number[];
  eps: number[];
  last: EvalResult | null;
}

interface Replay {
  frames: Frame[];
  i: number;
  timer: number;
  optimalSteps: number;
}

type Tool = 'shelf' | 'item' | 'dock' | 'start';

const initial = presetById('classic');

const S = {
  presetId: initial.id as string,
  layout: structuredClone(initial.layout) as Layout,
  start: [0, 0] as Cell,
  stochastic: false,
  slip: 0.1,
  maxSteps: initial.maxSteps,
  algo: 'q' as Algo,
  hyper: { ...initial.hyper } as Hyper,
  target: initial.episodes,
  speedIdx: 3,
  seed: 42,
  tool: 'shelf' as Tool,
  phase: 0 as 0 | 1,
  showHeat: true,
  showPolicy: true,
  showOptimal: false,
  tab: 'live' as 'live' | 'bench' | 'about',
  running: false,
  hover: null as Cell | null,
  env: null as unknown as WarehouseEnv,
  agent: null as unknown as TabularAgent,
  plan: null as unknown as Plan,
  problem: null as string | null,
  optMeanReturn: 0,
  history: null as unknown as History,
  replay: null as Replay | null,
  benchResults: null as BenchmarkRun[] | null,
  benchRunning: false,
  benchCancel: false,
  benchEpisodes: 1500,
  benchSeeds: 8,
};

// ---------------------------------------------------------------- build / reset
function rebuild(): void {
  stopTraining();
  stopReplay(true);
  S.plan = solve(S.layout);
  S.problem = validateLayout(S.layout);
  const sd = S.plan.dist[encode(S.layout.size, S.start[0], S.start[1], 0)];
  if (!S.problem && !Number.isFinite(sd)) S.problem = 'There is no route from the start cell to the item and the dock. Move a shelf or the start.';

  const rng = mulberry32(S.seed);
  S.env = new WarehouseEnv(S.layout, S.stochastic, rng, S.maxSteps, S.slip);
  S.agent = new TabularAgent(S.env.nStates, S.algo, S.hyper, rng);

  let sum = 0;
  let n = 0;
  for (const [x, y] of S.env.validStarts()) {
    const d = S.plan.dist[encode(S.layout.size, x, y, 0)];
    if (Number.isFinite(d)) {
      sum += 125 - d;
      n++;
    }
  }
  S.optMeanReturn = n ? sum / n : 0;

  S.history = { rewards: [], evalX: [], success: [], efficiency: [], eps: [], last: null };
  if (!S.problem) recordEval();
  syncUI();
  renderAll();
}

function recordEval(): void {
  const ev = evaluate(S.agent, S.layout, S.plan, S.stochastic, S.maxSteps, S.slip);
  const h = S.history;
  h.evalX.push(S.agent.episodes);
  h.success.push(ev.successRate * 100);
  h.efficiency.push(ev.efficiency * 100);
  h.eps.push(S.agent.eps * 100);
  h.last = ev;
}

// ---------------------------------------------------------------- training loop
function startTraining(): void {
  if (S.problem || S.running) return;
  if (S.agent.episodes >= S.target) S.target = Math.min(20000, S.agent.episodes + 1000);
  S.running = true;
  stopReplay(true);
  syncUI();
  requestAnimationFrame(trainFrame);
}

function stopTraining(): void {
  S.running = false;
}

function trainFrame(): void {
  if (!S.running) return;
  const t0 = performance.now();
  const perFrame = SPEEDS[S.speedIdx];
  const evalEvery = Math.max(10, Math.round(S.target / 200));
  let n = 0;
  while (n < perFrame && S.agent.episodes < S.target && performance.now() - t0 < 14) {
    S.history.rewards.push(S.agent.trainEpisode(S.env).reward);
    if (S.agent.episodes % evalEvery === 0) recordEval();
    n++;
  }
  if (S.agent.episodes >= S.target) {
    if (S.history.evalX[S.history.evalX.length - 1] !== S.agent.episodes) recordEval();
    S.running = false;
  }
  syncUI();
  renderAll();
  if (S.running) requestAnimationFrame(trainFrame);
}

// ---------------------------------------------------------------- replay
function idlePose(): RobotPose {
  return { x: S.start[0], y: S.start[1], carrying: 0, dir: 'down', collided: false, delivered: false };
}

function currentPose(): RobotPose {
  if (!S.replay) return idlePose();
  const f = S.replay.frames[S.replay.i];
  return { x: f.x, y: f.y, carrying: f.carrying, dir: f.dir, collided: f.collided, delivered: f.delivered };
}

function stopReplay(clear: boolean): void {
  if (S.replay) window.clearInterval(S.replay.timer);
  if (clear) {
    S.replay = null;
    setStatus('');
  }
}

function startReplay(): void {
  if (S.problem) return;
  stopTraining();
  stopReplay(true);
  const frames = rollout(S.agent, S.layout, S.stochastic, S.start, mulberry32((Math.random() * 1e9) | 0), S.maxSteps, S.slip);
  const optimalSteps = S.plan.dist[encode(S.layout.size, S.start[0], S.start[1], 0)];
  const replay: Replay = { frames, i: 0, optimalSteps, timer: 0 };
  S.replay = replay;
  replay.timer = window.setInterval(() => {
    replay.i++;
    const f = replay.frames[replay.i];
    if (replay.i >= replay.frames.length - 1) {
      window.clearInterval(replay.timer);
      const steps = replay.frames.length - 1;
      if (f.delivered) {
        setStatus(`Delivered in ${steps} steps (optimal: ${optimalSteps}). Return ${f.total}.`, 'good');
      } else {
        setStatus(`Did not deliver within ${steps} steps. The policy is not trained enough, or it is stuck in a loop.`, 'bad');
      }
    } else {
      setStatus(`Step ${replay.i}: ${ACTIONS[f.action]}. Return ${f.total}.`);
    }
    syncUI();
    renderAll();
  }, 170);
  syncUI();
  renderAll();
}

function setStatus(text: string, tone: '' | 'good' | 'bad' = ''): void {
  const el = $('replayStatus');
  el.textContent = text;
  el.className = `status ${tone}`.trim();
}

// ---------------------------------------------------------------- rendering
const view = (): ViewState => ({
  layout: S.layout,
  agent: S.agent,
  phase: S.phase,
  showHeat: S.showHeat,
  showPolicy: S.showPolicy,
  showOptimal: S.showOptimal,
  optimalPath: S.showOptimal && !S.problem ? optimalPath(S.layout, S.plan, S.start) : [],
  trail: S.replay ? S.replay.frames.slice(0, S.replay.i + 1).map((f) => [f.x, f.y] as Cell) : [],
  robot: currentPose(),
  start: S.start,
  hover: S.hover,
});

function renderAll(): void {
  renderGrid($<HTMLCanvasElement>('grid'), view());
  renderInfo();
  if (S.tab === 'live') {
    renderMetrics();
    renderCharts();
  } else if (S.tab === 'bench') renderBench();
}

function renderInfo(): void {
  const el = $('info');
  const c = S.hover;
  if (!c) {
    el.textContent = 'Hover a cell to inspect what the agent has learned.';
    return;
  }
  const n = S.layout.size;
  if (shelfSet(S.layout).has(c[1] * n + c[0])) {
    el.textContent = `Cell (${c[0]}, ${c[1]}): shelf`;
    return;
  }
  const desc = (phase: 0 | 1): string => {
    const s = encode(n, c[0], c[1], phase);
    if (S.agent.untouched(s)) return 'unexplored';
    const a = S.agent.greedy(s);
    return `${ARROWS[a]} (Q = ${S.agent.value(s, a).toFixed(1)})`;
  };
  const tag = sameCell(c, S.layout.item) ? ' [item]' : sameCell(c, S.layout.dock) ? ' [dock]' : '';
  el.textContent = `Cell (${c[0]}, ${c[1]})${tag}. Searching: ${desc(0)}. Carrying: ${desc(1)}.`;
}

function metric(label: string, value: string, sub = ''): string {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong><small>${sub || '&nbsp;'}</small></div>`;
}

function renderMetrics(): void {
  const h = S.history;
  const recent = h.rewards.slice(-50);
  const avg = recent.length ? mean(recent) : null;
  const ev = h.last;
  $('metrics').innerHTML = [
    metric('Episode', S.agent.episodes.toLocaleString(), `of ${S.target.toLocaleString()}`),
    metric('Exploration', S.agent.eps.toFixed(3), S.agent.eps <= S.hyper.epsMin + 1e-9 ? 'ε at minimum' : 'ε decaying'),
    metric('Avg reward', avg === null ? '–' : avg.toFixed(1), `optimal ≈ ${S.optMeanReturn.toFixed(0)}`),
    metric('Success', ev ? `${(ev.successRate * 100).toFixed(0)}%` : '–', 'all start cells'),
    metric('Efficiency', ev && ev.successRate > 0 ? `${(ev.efficiency * 100).toFixed(0)}%` : '–', 'vs optimal route'),
    metric(
      'Avg steps',
      ev && ev.successRate > 0 ? ev.meanSteps.toFixed(1) : '–',
      ev && ev.successRate > 0 ? `optimal ${ev.meanOptimalSteps.toFixed(1)}` : '',
    ),
  ].join('');
}

function legend(el: HTMLElement, items: [string, string][]): void {
  el.innerHTML = items.map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join('');
}

function renderCharts(): void {
  const h = S.history;
  const xMax = Math.max(S.target, 1);

  const raw = downsample(h.rewards, 500);
  const ma = movingAverage(h.rewards, 50);
  const maDs = downsample(ma, 500);
  const series: Series[] = [
    { label: 'Episode reward', color: C.main() + '55', data: raw.values, xs: raw.index.map((i) => i + 1), width: 1 },
    { label: 'Moving average', color: C.main(), data: maDs.values, xs: maDs.index.map((i) => i + 1) },
  ];
  drawChart($<HTMLCanvasElement>('chartReward'), {
    series,
    xMax,
    xLabel: 'episode',
    yMin: Math.min(-50, ...h.rewards.slice(-2000), 0),
    ref: S.optMeanReturn ? { y: S.optMeanReturn, label: 'optimal', color: C.good() } : undefined,
  });
  legend($('legendReward'), [
    [C.main() + '55', 'each episode'],
    [C.main(), 'moving average (50)'],
    [C.good(), S.stochastic ? 'optimal (no slip)' : 'optimal'],
  ]);

  drawChart($<HTMLCanvasElement>('chartQuality'), {
    series: [
      { label: 'Success', color: C.good(), data: h.success, xs: h.evalX },
      { label: 'Efficiency', color: C.warm(), data: h.efficiency, xs: h.evalX },
      { label: 'Exploration', color: C.muted(), data: h.eps, xs: h.evalX, dashed: true, width: 1.5 },
    ],
    xMax,
    xLabel: 'episode',
    yMin: 0,
    yMax: 100,
    yFormat: (v) => `${Math.round(v)}%`,
  });
  legend($('legendQuality'), [
    [C.good(), 'success rate'],
    [C.warm(), 'efficiency vs optimal'],
    [C.muted(), 'ε (exploration)'],
  ]);
}

// ---------------------------------------------------------------- benchmark
function renderBench(): void {
  const res = S.benchResults;
  $('benchResults').hidden = !res;
  if (!res) return;

  const win = 50;
  const series: Series[] = res.map((r) => {
    const curves = r.rewards.map((rw) => movingAverage(rw, win));
    const len = curves[0].length;
    const m: number[] = [];
    const lo: number[] = [];
    const hi: number[] = [];
    for (let i = 0; i < len; i++) {
      const col = curves.map((c) => c[i]);
      const mu = mean(col);
      const sd = std(col);
      m.push(mu);
      lo.push(mu - sd);
      hi.push(mu + sd);
    }
    const d = downsample(m, 300);
    const color = algoColor(r.algo);
    return {
      label: ALGO_INFO[r.algo].name,
      color,
      data: d.values,
      xs: d.index.map((i) => i + 1),
      band: { lo: d.index.map((i) => lo[i]), hi: d.index.map((i) => hi[i]) },
    };
  });
  drawChart($<HTMLCanvasElement>('chartBench'), {
    series,
    xMax: res[0].rewards[0].length,
    xLabel: 'episode',
    yMin: Math.max(-200, Math.min(-50, ...series.map((s) => Math.min(...(s.band?.lo ?? s.data))))),
    ref: S.optMeanReturn ? { y: S.optMeanReturn, label: 'optimal', color: C.muted() } : undefined,
  });
  legend($('legendBench'), res.map((r) => [algoColor(r.algo), ALGO_INFO[r.algo].name]));

  const threshold = 0.85 * S.optMeanReturn;
  const rows = res.map((r) => {
    const finals = r.rewards.map((rw) => mean(rw.slice(-100)));
    const reached = r.rewards
      .map((rw) => movingAverage(rw, win).findIndex((v) => v >= threshold) + 1)
      .filter((i) => i > 0);
    return {
      algo: r.algo,
      reward: mean(finals),
      rewardSd: std(finals),
      success: mean(r.finalSuccess) * 100,
      eff: mean(r.finalEfficiency) * 100,
      toThreshold: reached.length ? mean(reached) : Infinity,
      reachedCount: reached.length,
      seeds: r.rewards.length,
    };
  });
  const best = {
    reward: Math.max(...rows.map((r) => r.reward)),
    success: Math.max(...rows.map((r) => r.success)),
    eff: Math.max(...rows.map((r) => r.eff)),
    to: Math.min(...rows.map((r) => r.toThreshold)),
  };
  const cls = (cond: boolean) => (cond ? ' class="best"' : '');
  $('benchTable').innerHTML =
    `<thead><tr><th>Algorithm</th><th>Final reward</th><th>Success</th><th>Effi&shy;ciency</th><th>Eps to 85%</th></tr></thead><tbody>` +
    rows
      .map(
        (r) =>
          `<tr><td><i style="display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px;background:${algoColor(r.algo)}"></i>${r.algo === 'double' ? 'Double Q' : ALGO_INFO[r.algo].name}</td>` +
          `<td${cls(r.reward.toFixed(1) === best.reward.toFixed(1))}>${r.reward.toFixed(1)} ± ${r.rewardSd.toFixed(1)}</td>` +
          `<td${cls(Math.round(r.success) === Math.round(best.success))}>${r.success.toFixed(0)}%</td>` +
          `<td${cls(Math.round(r.eff) === Math.round(best.eff))}>${r.eff.toFixed(0)}%</td>` +
          `<td${cls(r.toThreshold === best.to)}>${
            Number.isFinite(r.toThreshold)
              ? Math.round(r.toThreshold).toLocaleString() + (r.reachedCount < r.seeds ? ` (${r.reachedCount}/${r.seeds})` : '')
              : '–'
          }</td></tr>`,
      )
      .join('') +
    '</tbody>';
  $('benchNote').textContent =
    `Mean over ${res[0].rewards.length} seeds. Final reward is the mean of the last 100 training episodes. ` +
    `"Eps to 85%" is the number of episodes until the 50-episode moving average first reaches 85% of the optimal return (${S.optMeanReturn.toFixed(0)})` +
    (S.stochastic ? ', which slip can make unreachable on a stochastic floor.' : '.');
}

async function benchmark(): Promise<void> {
  if (S.benchRunning) {
    S.benchCancel = true;
    return;
  }
  if (S.problem) return;
  stopTraining();
  S.benchRunning = true;
  S.benchCancel = false;
  const btn = $<HTMLButtonElement>('runBench');
  btn.textContent = 'Cancel';
  $('benchProgress').hidden = false;
  $('benchStatus').textContent = 'Training 4 algorithms…';
  const t0 = performance.now();
  const results = await runBenchmark(
    {
      layout: S.layout,
      stochastic: S.stochastic,
      maxSteps: S.maxSteps,
      slip: S.slip,
      hyper: { ...S.hyper },
      episodes: S.benchEpisodes,
      seeds: S.benchSeeds,
      algos: ['q', 'sarsa', 'expected', 'double'],
    },
    (p) => (($('benchBar').style.width = `${(p * 100).toFixed(0)}%`), undefined),
    () => S.benchCancel,
  );
  S.benchRunning = false;
  btn.textContent = 'Run benchmark';
  $('benchProgress').hidden = true;
  if (S.benchCancel || results.length < 4) {
    $('benchStatus').textContent = 'Cancelled.';
    return;
  }
  S.benchResults = results;
  $('benchStatus').textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
  renderBench();
}

// ---------------------------------------------------------------- UI sync
function syncUI(): void {
  $('sizeVal').textContent = `${S.layout.size}×${S.layout.size}`;
  ($('size') as HTMLInputElement).value = String(S.layout.size);
  $('maxStepsVal').textContent = String(S.maxSteps);
  ($('maxSteps') as HTMLInputElement).value = String(S.maxSteps);
  ($('stochastic') as HTMLInputElement).checked = S.stochastic;
  $('slipField').hidden = !S.stochastic;
  $('slipVal').textContent = `${Math.round(S.slip * 100)}%`;
  ($('slip') as HTMLInputElement).value = String(Math.round(S.slip * 100));
  ($('preset') as HTMLSelectElement).value = S.presetId;
  $('presetBlurb').textContent = S.presetId === 'custom' ? 'Your own layout.' : presetById(S.presetId).blurb;
  ($('algo') as HTMLSelectElement).value = S.algo;
  $('algoBlurb').textContent = ALGO_INFO[S.algo].blurb;
  $('alphaVal').textContent = S.hyper.alpha.toFixed(2);
  ($('alpha') as HTMLInputElement).value = String(S.hyper.alpha);
  $('gammaVal').textContent = S.hyper.gamma.toFixed(3);
  ($('gamma') as HTMLInputElement).value = String(S.hyper.gamma);
  $('decayVal').textContent = S.hyper.epsDecay.toFixed(4);
  ($('decay') as HTMLInputElement).value = String(S.hyper.epsDecay);
  $('episodesVal').textContent = S.target.toLocaleString();
  ($('episodes') as HTMLInputElement).value = String(S.target);
  $('speedVal').textContent = `×${SPEEDS[S.speedIdx]}`;
  ($('speed') as HTMLInputElement).value = String(S.speedIdx);
  ($('seed') as HTMLInputElement).value = String(S.seed);

  $('bEpisodesVal').textContent = S.benchEpisodes.toLocaleString();
  ($('bEpisodes') as HTMLInputElement).value = String(S.benchEpisodes);
  $('bSeedsVal').textContent = String(S.benchSeeds);
  ($('bSeeds') as HTMLInputElement).value = String(S.benchSeeds);

  document.querySelectorAll<HTMLButtonElement>('.tool').forEach((b) => {
    const on = b.dataset.tool === S.tool;
    b.classList.toggle('active', on);
    b.setAttribute('aria-checked', String(on));
  });
  const flag = (id: string, on: boolean) => {
    $(id).classList.toggle('on', on);
    $(id).setAttribute('aria-pressed', String(on));
  };
  flag('tHeat', S.showHeat);
  flag('tPolicy', S.showPolicy);
  flag('tOptimal', S.showOptimal);
  for (const p of [0, 1]) {
    $(`phase${p}`).classList.toggle('active', S.phase === p);
    $(`phase${p}`).setAttribute('aria-pressed', String(S.phase === p));
  }

  const problem = $('problem');
  problem.hidden = !S.problem;
  problem.textContent = S.problem ?? '';

  const finished = S.agent.episodes >= S.target;
  const train = $<HTMLButtonElement>('train');
  train.textContent = S.running ? 'Pause' : S.agent.episodes === 0 ? 'Train' : finished ? 'Train more' : 'Resume';
  train.disabled = !!S.problem;
  $<HTMLButtonElement>('watch').disabled = !!S.problem;
  $<HTMLButtonElement>('randomStart').disabled = !!S.problem;
  $<HTMLButtonElement>('runBench').disabled = !!S.problem && !S.benchRunning;
}

function setTab(tab: typeof S.tab): void {
  S.tab = tab;
  document.querySelectorAll<HTMLButtonElement>('.tab').forEach((b) => {
    const on = b.dataset.tab === tab;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', String(on));
  });
  $('view-live').hidden = tab !== 'live';
  $('view-bench').hidden = tab !== 'bench';
  $('view-about').hidden = tab !== 'about';
  renderAll();
}

// ---------------------------------------------------------------- editing
function applyPreset(id: string): void {
  const p = presetById(id);
  S.presetId = p.id;
  S.layout = structuredClone(p.layout);
  S.maxSteps = p.maxSteps;
  S.target = p.episodes;
  Object.assign(S.hyper, p.hyper);
  S.start = firstFree(S.layout);
  S.benchEpisodes = Math.min(6000, p.episodes);
  S.benchResults = null;
  rebuild();
}

function firstFree(layout: Layout): Cell {
  const sh = shelfSet(layout);
  for (let y = 0; y < layout.size; y++)
    for (let x = 0; x < layout.size; x++)
      if (!sh.has(y * layout.size + x) && !sameCell([x, y], layout.item) && !sameCell([x, y], layout.dock)) return [x, y];
  return [0, 0];
}

function markCustom(): void {
  S.presetId = 'custom';
  S.benchResults = null;
}

function applyTool(c: Cell, mode: 'add' | 'remove'): boolean {
  const L = S.layout;
  const isShelf = L.shelves.some((s) => sameCell(s, c));
  switch (S.tool) {
    case 'shelf':
      if (mode === 'add') {
        if (isShelf || sameCell(c, L.item) || sameCell(c, L.dock) || sameCell(c, S.start)) return false;
        L.shelves.push([c[0], c[1]]);
      } else {
        if (!isShelf) return false;
        L.shelves = L.shelves.filter((s) => !sameCell(s, c));
      }
      return true;
    case 'item':
      if (isShelf || sameCell(c, L.dock) || sameCell(c, L.item)) return false;
      L.item = [c[0], c[1]];
      return true;
    case 'dock':
      if (isShelf || sameCell(c, L.item) || sameCell(c, L.dock)) return false;
      L.dock = [c[0], c[1]];
      return true;
    case 'start':
      if (isShelf || sameCell(c, S.start)) return false;
      S.start = [c[0], c[1]];
      return true;
  }
}

function resizeGrid(size: number): void {
  const L = S.layout;
  const clamp = (c: Cell): Cell => [Math.min(c[0], size - 1), Math.min(c[1], size - 1)];
  L.size = size;
  L.shelves = L.shelves.filter(([x, y]) => x < size && y < size);
  L.item = clamp(L.item);
  L.dock = clamp(L.dock);
  if (sameCell(L.item, L.dock)) L.item = [0, size - 1];
  L.shelves = L.shelves.filter((s) => !sameCell(s, L.item) && !sameCell(s, L.dock));
  S.start = clamp(S.start);
  L.shelves = L.shelves.filter((s) => !sameCell(s, S.start));
  markCustom();
  rebuild();
}

// ---------------------------------------------------------------- wiring
function wire(): void {
  const grid = $<HTMLCanvasElement>('grid');
  let painting: 'add' | 'remove' | null = null;
  let last: Cell | null = null;

  const paint = (e: PointerEvent) => {
    const c = cellFromEvent(grid, e, S.layout.size);
    S.hover = c;
    if (painting && c && !(last && sameCell(last, c))) {
      last = c;
      if (applyTool(c, painting)) {
        markCustom();
        rebuild();
        return;
      }
    }
    renderAll();
  };
  grid.addEventListener('pointerdown', (e) => {
    const c = cellFromEvent(grid, e, S.layout.size);
    if (!c) return;
    grid.setPointerCapture(e.pointerId);
    painting = S.tool === 'shelf' && S.layout.shelves.some((s) => sameCell(s, c)) ? 'remove' : 'add';
    last = null;
    paint(e);
  });
  grid.addEventListener('pointermove', paint);
  grid.addEventListener('pointerup', () => (painting = null));
  grid.addEventListener('pointercancel', () => (painting = null));
  grid.addEventListener('pointerleave', () => {
    S.hover = null;
    painting = null;
    renderAll();
  });

  const preset = $<HTMLSelectElement>('preset');
  preset.innerHTML = [...PRESETS.map((p) => `<option value="${p.id}">${p.name}</option>`), '<option value="custom">Custom</option>'].join('');
  preset.addEventListener('change', () => {
    if (preset.value !== 'custom') applyPreset(preset.value);
  });

  const algo = $<HTMLSelectElement>('algo');
  algo.innerHTML = (Object.keys(ALGO_INFO) as Algo[]).map((a) => `<option value="${a}">${ALGO_INFO[a].name}</option>`).join('');
  algo.addEventListener('change', () => {
    S.algo = algo.value as Algo;
    rebuild();
  });

  const range = (id: string, fn: (v: number) => void, rebuilds = false) =>
    $<HTMLInputElement>(id).addEventListener('input', (e) => {
      fn(Number((e.target as HTMLInputElement).value));
      if (rebuilds) rebuild();
      else {
        syncUI();
        renderAll();
      }
    });
  range('size', (v) => resizeGrid(v));
  range('maxSteps', (v) => (S.maxSteps = v), true);
  range('slip', (v) => (S.slip = v / 100), true);
  range('alpha', (v) => (S.hyper.alpha = v));
  range('gamma', (v) => (S.hyper.gamma = v));
  range('decay', (v) => (S.hyper.epsDecay = v));
  range('episodes', (v) => (S.target = v));
  range('speed', (v) => (S.speedIdx = v));
  range('bEpisodes', (v) => (S.benchEpisodes = v));
  range('bSeeds', (v) => (S.benchSeeds = v));

  $<HTMLInputElement>('stochastic').addEventListener('change', (e) => {
    S.stochastic = (e.target as HTMLInputElement).checked;
    S.benchResults = null;
    rebuild();
  });
  $<HTMLInputElement>('seed').addEventListener('change', (e) => {
    S.seed = Math.max(0, Math.floor(Number((e.target as HTMLInputElement).value) || 0));
    rebuild();
  });

  document.querySelectorAll<HTMLButtonElement>('.tool').forEach((b) =>
    b.addEventListener('click', () => {
      S.tool = b.dataset.tool as Tool;
      syncUI();
    }),
  );
  $('clearShelves').addEventListener('click', () => {
    S.layout.shelves = [];
    markCustom();
    rebuild();
  });

  const flip = (id: string, key: 'showHeat' | 'showPolicy' | 'showOptimal') =>
    $(id).addEventListener('click', () => {
      S[key] = !S[key];
      syncUI();
      renderAll();
    });
  flip('tHeat', 'showHeat');
  flip('tPolicy', 'showPolicy');
  flip('tOptimal', 'showOptimal');
  for (const p of [0, 1] as const)
    $(`phase${p}`).addEventListener('click', () => {
      S.phase = p;
      syncUI();
      renderAll();
    });

  $('train').addEventListener('click', () => (S.running ? (stopTraining(), syncUI()) : startTraining()));
  $('reset').addEventListener('click', rebuild);
  $('watch').addEventListener('click', startReplay);
  $('randomStart').addEventListener('click', () => {
    const cells = S.env.validStarts().filter((c) => Number.isFinite(S.plan.dist[encode(S.layout.size, c[0], c[1], 0)]));
    S.start = cells[Math.floor(Math.random() * cells.length)];
    startReplay();
  });
  $('runBench').addEventListener('click', () => void benchmark());

  document.querySelectorAll<HTMLButtonElement>('.tab').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab as typeof S.tab)));

  window.addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement;
    if (e.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(t.tagName)) {
      e.preventDefault();
      if (S.tab === 'live') $('train').click();
    }
  });
  new ResizeObserver(() => renderAll()).observe(document.body);
}

// ---------------------------------------------------------------- boot
const themeSel = $<HTMLSelectElement>('themeSel');
themeSel.value = document.documentElement.dataset.theme ?? THEMES[0];
themeSel.addEventListener('change', () => {
  document.documentElement.dataset.theme = themeSel.value;
  const url = new URL(location.href);
  url.searchParams.set('theme', themeSel.value);
  history.replaceState(null, '', url);
  try {
    localStorage.setItem('rl-lab-theme', themeSel.value);
  } catch {
    /* ignore: the choice just will not persist */
  }
  requestAnimationFrame(() => renderAll());
});
wire();
loadSprites(() => renderAll());
S.start = firstFree(S.layout);
rebuild();
setTab('live');

// Exposed for tests and quick poking from the dev console.
(window as unknown as { __rl: typeof S }).__rl = S;
