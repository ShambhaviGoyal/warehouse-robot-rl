import { TabularAgent, Algo, Hyper } from './agents';
import { Cell, Dir, Layout, Rng, WarehouseEnv, encode, mulberry32 } from './env';
import { Plan, solve } from './planner';

export interface Frame {
  x: number;
  y: number;
  carrying: number;
  dir: Dir;
  collided: boolean;
  delivered: boolean;
  reward: number;
  total: number;
  action: number;
}

/** Follow the greedy policy from a start cell and record every frame (for the replay). */
export function rollout(
  agent: TabularAgent,
  layout: Layout,
  stochastic: boolean,
  start: Cell,
  rng: Rng,
  maxSteps: number,
  slip: number,
): Frame[] {
  const env = new WarehouseEnv(layout, stochastic, rng, maxSteps, slip);
  env.reset(start);
  let total = 0;
  const frames: Frame[] = [
    { x: env.x, y: env.y, carrying: 0, dir: 'down', collided: false, delivered: false, reward: 0, total: 0, action: -1 },
  ];
  for (;;) {
    const a = agent.greedy(env.state);
    const r = env.step(a);
    total += r.reward;
    frames.push({
      x: env.x,
      y: env.y,
      carrying: env.carrying,
      dir: env.lastDir,
      collided: r.collided,
      delivered: env.delivered,
      reward: r.reward,
      total,
      action: a,
    });
    if (r.terminated || r.truncated) break;
  }
  return frames;
}

export interface EvalResult {
  successRate: number;
  /** Mean over successful runs of optimalSteps / actualSteps. 1 means perfectly optimal. */
  efficiency: number;
  meanSteps: number;
  meanOptimalSteps: number;
}

/** Greedy evaluation from every legal start cell. */
export function evaluate(
  agent: TabularAgent,
  layout: Layout,
  plan: Plan,
  stochastic: boolean,
  maxSteps: number,
  slip: number,
  seed = 12345,
): EvalResult {
  const rng = mulberry32(seed);
  const env = new WarehouseEnv(layout, stochastic, rng, maxSteps, slip);
  const starts = env.validStarts();
  const trials = stochastic ? 4 : 1;
  let runs = 0;
  let ok = 0;
  let effSum = 0;
  let stepSum = 0;
  let optSum = 0;

  for (const [x, y] of starts) {
    const opt = plan.dist[encode(layout.size, x, y, 0)];
    if (!Number.isFinite(opt)) continue; // walled-off pockets can never deliver
    for (let t = 0; t < trials; t++) {
      env.reset([x, y]);
      let terminated = false;
      for (;;) {
        const r = env.step(agent.greedy(env.state));
        if (r.terminated) terminated = true;
        if (r.terminated || r.truncated) break;
      }
      runs++;
      if (terminated) {
        ok++;
        effSum += opt / env.steps;
        stepSum += env.steps;
        optSum += opt;
      }
    }
  }
  return {
    successRate: runs ? ok / runs : 0,
    efficiency: ok ? effSum / ok : 0,
    meanSteps: ok ? stepSum / ok : 0,
    meanOptimalSteps: ok ? optSum / ok : 0,
  };
}

/** Moving average with a trailing window. */
export function movingAverage(xs: ArrayLike<number>, window: number): number[] {
  const out: number[] = new Array(xs.length);
  let sum = 0;
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i];
    if (i >= window) sum -= xs[i - window];
    out[i] = sum / Math.min(i + 1, window);
  }
  return out;
}

export interface BenchmarkRun {
  algo: Algo;
  /** Per-seed raw episode rewards. */
  rewards: number[][];
  finalSuccess: number[];
  finalEfficiency: number[];
}

export interface BenchmarkConfig {
  layout: Layout;
  stochastic: boolean;
  maxSteps: number;
  slip: number;
  hyper: Hyper;
  episodes: number;
  seeds: number;
  algos: Algo[];
}

/**
 * Trains every algorithm on several seeds. Yields to the browser between runs
 * so the page stays responsive, and reports progress in [0, 1].
 */
export async function runBenchmark(
  cfg: BenchmarkConfig,
  onProgress: (p: number) => void,
  isCancelled: () => boolean,
): Promise<BenchmarkRun[]> {
  const plan = solve(cfg.layout);
  const results: BenchmarkRun[] = [];
  const totalRuns = cfg.algos.length * cfg.seeds;
  let done = 0;

  for (const algo of cfg.algos) {
    const run: BenchmarkRun = { algo, rewards: [], finalSuccess: [], finalEfficiency: [] };
    for (let seed = 0; seed < cfg.seeds; seed++) {
      if (isCancelled()) return results;
      const rng = mulberry32(1000 + seed * 7919);
      const env = new WarehouseEnv(cfg.layout, cfg.stochastic, rng, cfg.maxSteps, cfg.slip);
      const agent = new TabularAgent(env.nStates, algo, cfg.hyper, rng);
      const rewards: number[] = [];
      for (let e = 0; e < cfg.episodes; e++) rewards.push(agent.trainEpisode(env).reward);
      const ev = evaluate(agent, cfg.layout, plan, cfg.stochastic, cfg.maxSteps, cfg.slip);
      run.rewards.push(rewards);
      run.finalSuccess.push(ev.successRate);
      run.finalEfficiency.push(ev.efficiency);
      done++;
      onProgress(done / totalRuns);
      await new Promise((r) => setTimeout(r, 0));
    }
    results.push(run);
  }
  return results;
}

export const mean = (xs: number[]): number => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
export const std = (xs: number[]): number => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};
