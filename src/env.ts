// Warehouse robot environment.
// Rules mirror the original Gymnasium environment: +100 delivery, +25 pickup,
// -1 per step, -20 extra for bumping a shelf, 10% slip on movement actions only.

export type Cell = [number, number];

export interface Layout {
  size: number;
  shelves: Cell[];
  item: Cell;
  dock: Cell;
}

export const ACTIONS = ['up', 'down', 'left', 'right', 'pick', 'drop'] as const;
export const N_ACTIONS = ACTIONS.length;
export const DIRS = ['up', 'down', 'left', 'right'] as const;
export type Dir = (typeof DIRS)[number];

export const REWARDS = { delivery: 100, pickup: 25, step: -1, crash: -20 };
export const DEFAULT_MAX_STEPS = 100;

export const CLASSIC_LAYOUT: Layout = {
  size: 6,
  shelves: [
    [2, 2],
    [2, 3],
    [3, 2],
    [3, 3],
  ],
  item: [0, 4],
  dock: [5, 5],
};

export type Rng = () => number;

/** Small seeded PRNG so every run is reproducible. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const encode = (size: number, x: number, y: number, carrying: number): number =>
  (y * size + x) * 2 + carrying;

export const decode = (size: number, s: number): { x: number; y: number; carrying: number } => {
  const carrying = s % 2;
  const cell = (s - carrying) / 2;
  return { x: cell % size, y: Math.floor(cell / size), carrying };
};

export const sameCell = (a: Cell, b: Cell): boolean => a[0] === b[0] && a[1] === b[1];

export function shelfSet(layout: Layout): Set<number> {
  return new Set(layout.shelves.map(([x, y]) => y * layout.size + x));
}

export interface Transition {
  x: number;
  y: number;
  carrying: number;
  reward: number;
  done: boolean;
  collided: boolean;
}

const MOVES: Cell[] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

/** Pure, deterministic dynamics. The simulator and the optimal planner both use this. */
export function transition(
  layout: Layout,
  shelves: Set<number>,
  x: number,
  y: number,
  carrying: number,
  action: number,
): Transition {
  let reward = REWARDS.step;
  let done = false;
  let collided = false;
  let nx = x;
  let ny = y;
  let nc = carrying;

  if (action < 4) {
    const tx = x + MOVES[action][0];
    const ty = y + MOVES[action][1];
    if (tx < 0 || ty < 0 || tx >= layout.size || ty >= layout.size) {
      collided = true; // wall bump: no extra penalty, no movement
    } else if (shelves.has(ty * layout.size + tx)) {
      collided = true;
      reward += REWARDS.crash;
    } else {
      nx = tx;
      ny = ty;
    }
  } else if (action === 4) {
    if (carrying === 0 && x === layout.item[0] && y === layout.item[1]) {
      nc = 1;
      reward += REWARDS.pickup;
    }
  } else if (action === 5) {
    if (carrying === 1 && x === layout.dock[0] && y === layout.dock[1]) {
      nc = 0;
      reward += REWARDS.delivery;
      done = true;
    }
  }
  return { x: nx, y: ny, carrying: nc, reward, done, collided };
}

export interface StepResult {
  state: number;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  collided: boolean;
}

export class WarehouseEnv {
  readonly shelves: Set<number>;
  x = 0;
  y = 0;
  carrying = 0;
  steps = 0;
  delivered = false;
  lastDir: Dir = 'down';
  collided = false;

  constructor(
    public layout: Layout,
    public stochastic = false,
    public rng: Rng = Math.random,
    public maxSteps = DEFAULT_MAX_STEPS,
    public slip = 0.1,
  ) {
    this.shelves = shelfSet(layout);
  }

  get nStates(): number {
    return this.layout.size * this.layout.size * 2;
  }

  get state(): number {
    return encode(this.layout.size, this.x, this.y, this.carrying);
  }

  /** Every non-shelf cell is a legal starting cell. */
  validStarts(): Cell[] {
    const out: Cell[] = [];
    for (let y = 0; y < this.layout.size; y++)
      for (let x = 0; x < this.layout.size; x++)
        if (!this.shelves.has(y * this.layout.size + x)) out.push([x, y]);
    return out;
  }

  reset(start?: Cell): number {
    const cells = this.validStarts();
    const [sx, sy] = start ?? cells[Math.floor(this.rng() * cells.length)];
    this.x = sx;
    this.y = sy;
    this.carrying = 0;
    this.steps = 0;
    this.delivered = false;
    this.lastDir = 'down';
    this.collided = false;
    return this.state;
  }

  step(action: number): StepResult {
    this.steps += 1;
    if (action < 4) this.lastDir = DIRS[action];

    let t: Transition;
    if (this.stochastic && action < 4 && this.rng() < this.slip) {
      // Slip: the robot stays put, with no collision penalty.
      t = { x: this.x, y: this.y, carrying: this.carrying, reward: REWARDS.step, done: false, collided: false };
    } else {
      t = transition(this.layout, this.shelves, this.x, this.y, this.carrying, action);
    }
    this.x = t.x;
    this.y = t.y;
    this.carrying = t.carrying;
    this.collided = t.collided;
    if (t.done) this.delivered = true;

    return {
      state: this.state,
      reward: t.reward,
      terminated: t.done,
      truncated: !t.done && this.steps >= this.maxSteps,
      collided: t.collided,
    };
  }
}

/** Returns a human-readable problem with the layout, or null if it is valid. */
export function validateLayout(layout: Layout): string | null {
  const { size, item, dock } = layout;
  const shelves = shelfSet(layout);
  if (shelves.has(item[1] * size + item[0])) return 'The item is sitting on a shelf.';
  if (shelves.has(dock[1] * size + dock[0])) return 'The dock is sitting on a shelf.';
  if (sameCell(item, dock)) return 'The item and the dock are in the same cell.';
  return null;
}
