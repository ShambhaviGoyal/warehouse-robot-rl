import { N_ACTIONS, Rng, WarehouseEnv } from './env';

export type Algo = 'q' | 'sarsa' | 'expected' | 'double';

export const ALGO_INFO: Record<Algo, { name: string; blurb: string }> = {
  q: {
    name: 'Q-learning',
    blurb: 'Off-policy. Learns from the best next action, so it is optimistic about exploration.',
  },
  sarsa: {
    name: 'SARSA',
    blurb: 'On-policy. Learns from the action it actually takes next, so it accounts for its own exploration.',
  },
  expected: {
    name: 'Expected SARSA',
    blurb: 'Averages over the policy’s next-action probabilities. Lower variance than SARSA.',
  },
  double: {
    name: 'Double Q-learning',
    blurb: 'Two tables correct each other, which reduces the over-estimation bias of plain Q-learning.',
  },
};

export interface Hyper {
  alpha: number;
  gamma: number;
  epsStart: number;
  epsDecay: number;
  epsMin: number;
}

export const DEFAULT_HYPER: Hyper = {
  alpha: 0.1,
  gamma: 0.95,
  epsStart: 1.0,
  epsDecay: 0.995,
  epsMin: 0.01,
};

export interface EpisodeResult {
  reward: number;
  steps: number;
  delivered: boolean;
}

export class TabularAgent {
  q: Float64Array;
  q2: Float64Array | null;
  eps: number;
  episodes = 0;

  constructor(
    public nStates: number,
    public algo: Algo,
    public h: Hyper,
    public rng: Rng,
  ) {
    this.q = new Float64Array(nStates * N_ACTIONS);
    this.q2 = algo === 'double' ? new Float64Array(nStates * N_ACTIONS) : null;
    this.eps = h.epsStart;
  }

  /** The value the agent acts on (average of both tables for Double Q). */
  value(s: number, a: number): number {
    const i = s * N_ACTIONS + a;
    return this.q2 ? (this.q[i] + this.q2[i]) / 2 : this.q[i];
  }

  maxValue(s: number): number {
    let best = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) best = Math.max(best, this.value(s, a));
    return best;
  }

  /** Deterministic greedy action (first maximum), used for evaluation and display. */
  greedy(s: number): number {
    let best = 0;
    let bv = this.value(s, 0);
    for (let a = 1; a < N_ACTIONS; a++) {
      const v = this.value(s, a);
      if (v > bv) {
        bv = v;
        best = a;
      }
    }
    return best;
  }

  /** True when the agent has learned nothing about this state yet. */
  untouched(s: number): boolean {
    const first = this.value(s, 0);
    for (let a = 1; a < N_ACTIONS; a++) if (this.value(s, a) !== first) return false;
    return true;
  }

  /** Epsilon-greedy with random tie-breaking, so untrained states explore properly. */
  act(s: number): number {
    if (this.rng() < this.eps) return Math.floor(this.rng() * N_ACTIONS);
    let bv = -Infinity;
    const ties: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      const v = this.value(s, a);
      if (v > bv + 1e-12) {
        bv = v;
        ties.length = 0;
        ties.push(a);
      } else if (Math.abs(v - bv) <= 1e-12) ties.push(a);
    }
    return ties[Math.floor(this.rng() * ties.length)];
  }

  /** Expected value of the next state under the current epsilon-greedy policy. */
  private expectedValue(s: number): number {
    let bv = -Infinity;
    let count = 0;
    for (let a = 0; a < N_ACTIONS; a++) {
      const v = this.value(s, a);
      if (v > bv + 1e-12) {
        bv = v;
        count = 1;
      } else if (Math.abs(v - bv) <= 1e-12) count++;
    }
    let e = 0;
    for (let a = 0; a < N_ACTIONS; a++) {
      const v = this.value(s, a);
      const greedyShare = Math.abs(v - bv) <= 1e-12 ? (1 - this.eps) / count : 0;
      e += (this.eps / N_ACTIONS + greedyShare) * v;
    }
    return e;
  }

  private argmaxOf(table: Float64Array, s: number): number {
    let best = 0;
    let bv = table[s * N_ACTIONS];
    for (let a = 1; a < N_ACTIONS; a++) {
      const v = table[s * N_ACTIONS + a];
      if (v > bv) {
        bv = v;
        best = a;
      }
    }
    return best;
  }

  trainEpisode(env: WarehouseEnv): EpisodeResult {
    const { alpha, gamma } = this.h;
    let s = env.reset();
    let a = this.act(s);
    let total = 0;
    let steps = 0;
    let delivered = false;

    for (;;) {
      const r = env.step(a);
      const s2 = r.state;
      total += r.reward;
      steps++;
      const nextA = r.terminated ? 0 : this.act(s2);
      const i = s * N_ACTIONS + a;

      // Time-limit truncation still bootstraps; only true termination zeroes the future.
      if (this.algo === 'double' && this.q2) {
        const useFirst = this.rng() < 0.5;
        const [upd, other] = useFirst ? [this.q, this.q2] : [this.q2, this.q];
        const future = r.terminated ? 0 : other[s2 * N_ACTIONS + this.argmaxOf(upd, s2)];
        upd[i] += alpha * (r.reward + gamma * future - upd[i]);
      } else {
        let future = 0;
        if (!r.terminated) {
          if (this.algo === 'q') future = this.maxValue(s2);
          else if (this.algo === 'sarsa') future = this.value(s2, nextA);
          else future = this.expectedValue(s2);
        }
        this.q[i] += alpha * (r.reward + gamma * future - this.q[i]);
      }

      if (r.terminated) {
        delivered = true;
        break;
      }
      if (r.truncated) break;
      s = s2;
      a = nextA;
    }

    this.episodes++;
    this.eps = Math.max(this.h.epsMin, this.eps * this.h.epsDecay);
    return { reward: total, steps, delivered };
  }
}
