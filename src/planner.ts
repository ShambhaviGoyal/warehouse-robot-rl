import { Layout, N_ACTIONS, decode, encode, shelfSet, transition } from './env';

export interface Plan {
  /** Minimum number of actions to finish a delivery from each state (Infinity if impossible). */
  dist: Float64Array;
  /** An optimal first action from each state, or -1 if unreachable/terminal. */
  action: Int8Array;
}

/**
 * Exact shortest-path solution of the deterministic warehouse via value iteration
 * over the same transition function the simulator uses. Because every action costs -1
 * and crashing never helps, fewest steps is also the highest return: return = 125 - steps.
 */
export function solve(layout: Layout): Plan {
  const n = layout.size * layout.size * 2;
  const shelves = shelfSet(layout);
  const dist = new Float64Array(n).fill(Infinity);
  const action = new Int8Array(n).fill(-1);

  let changed = true;
  while (changed) {
    changed = false;
    for (let s = 0; s < n; s++) {
      const { x, y, carrying } = decode(layout.size, s);
      if (shelves.has(y * layout.size + x)) continue;
      for (let a = 0; a < N_ACTIONS; a++) {
        const t = transition(layout, shelves, x, y, carrying, a);
        if (t.collided) continue; // a bump never shortens a route
        const next = t.done ? 0 : dist[encode(layout.size, t.x, t.y, t.carrying)];
        if (next + 1 < dist[s]) {
          dist[s] = next + 1;
          action[s] = a;
          changed = true;
        }
      }
    }
  }
  return { dist, action };
}

/** Follow the optimal policy from a starting cell. Returns the visited cells (incl. start). */
export function optimalPath(layout: Layout, plan: Plan, start: [number, number]): [number, number][] {
  const shelves = shelfSet(layout);
  let x = start[0];
  let y = start[1];
  let c = 0;
  const path: [number, number][] = [[x, y]];
  for (let i = 0; i < 500; i++) {
    const a = plan.action[encode(layout.size, x, y, c)];
    if (a < 0) break;
    const t = transition(layout, shelves, x, y, c, a);
    x = t.x;
    y = t.y;
    c = t.carrying;
    if (path[path.length - 1][0] !== x || path[path.length - 1][1] !== y) path.push([x, y]);
    if (t.done) break;
  }
  return path;
}

/** Highest achievable return from a start cell (deterministic dynamics). */
export const optimalReturn = (plan: Plan, layout: Layout, x: number, y: number): number =>
  125 - plan.dist[encode(layout.size, x, y, 0)];

/** True when every non-shelf cell can finish a delivery. */
export function isSolvable(layout: Layout, plan: Plan): boolean {
  const shelves = shelfSet(layout);
  for (let y = 0; y < layout.size; y++)
    for (let x = 0; x < layout.size; x++) {
      if (shelves.has(y * layout.size + x)) continue;
      if (!Number.isFinite(plan.dist[encode(layout.size, x, y, 0)])) return false;
    }
  return true;
}
