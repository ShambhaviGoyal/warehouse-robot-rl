import { describe, expect, it } from 'vitest';
import { CLASSIC_LAYOUT, WarehouseEnv, mulberry32, shelfSet, transition, validateLayout } from '../src/env';
import { optimalPath, solve } from '../src/planner';

const UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, PICK = 4, DROP = 5;
const mk = (stochastic = false, seed = 1) => new WarehouseEnv(CLASSIC_LAYOUT, stochastic, mulberry32(seed));

describe('environment rules', () => {
  it('plain move costs -1 and moves the robot', () => {
    const e = mk();
    e.reset([0, 0]);
    const r = e.step(RIGHT);
    expect(r.reward).toBe(-1);
    expect([e.x, e.y]).toEqual([1, 0]);
  });

  it('up is y-1 and down is y+1', () => {
    const e = mk();
    e.reset([0, 2]);
    e.step(UP);
    expect(e.y).toBe(1);
    e.step(DOWN);
    e.step(DOWN);
    expect(e.y).toBe(3);
  });

  it('wall bump costs -1 and does not move', () => {
    const e = mk();
    e.reset([0, 0]);
    const r = e.step(LEFT);
    expect(r.reward).toBe(-1);
    expect(r.collided).toBe(true);
    expect([e.x, e.y]).toEqual([0, 0]);
  });

  it('shelf bump costs -21 and does not move', () => {
    const e = mk();
    e.reset([1, 2]);
    const r = e.step(RIGHT); // (2,2) is a shelf
    expect(r.reward).toBe(-21);
    expect([e.x, e.y]).toEqual([1, 2]);
  });

  it('pickup works only on the item cell', () => {
    const e = mk();
    e.reset([0, 0]);
    expect(e.step(PICK).reward).toBe(-1);
    expect(e.carrying).toBe(0);
    e.reset([0, 4]);
    expect(e.step(PICK).reward).toBe(24);
    expect(e.carrying).toBe(1);
    expect(e.step(PICK).reward).toBe(-1); // already carrying
  });

  it('drop works only at the dock while carrying, and ends the episode', () => {
    const e = mk();
    e.reset([5, 5]);
    expect(e.step(DROP).reward).toBe(-1); // not carrying
    e.carrying = 1;
    const r = e.step(DROP);
    expect(r.reward).toBe(99);
    expect(r.terminated).toBe(true);
    e.reset([0, 0]);
    e.carrying = 1;
    expect(e.step(DROP).terminated).toBe(false); // wrong cell
  });

  it('truncates after the step limit', () => {
    const e = mk();
    e.reset([0, 0]);
    let r = e.step(LEFT);
    for (let i = 1; i < 100; i++) r = e.step(LEFT);
    expect(r.truncated).toBe(true);
  });

  it('slip only affects movement and happens about 10% of the time', () => {
    const e = new WarehouseEnv({ ...CLASSIC_LAYOUT, shelves: [] }, true, mulberry32(7), 1e9);
    e.reset([0, 0]);
    let stayed = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
      e.x = 2;
      e.y = 2;
      e.step(RIGHT);
      if (e.x === 2) stayed++;
    }
    expect(stayed / n).toBeGreaterThan(0.08);
    expect(stayed / n).toBeLessThan(0.12);
    e.x = 0;
    e.y = 4;
    e.carrying = 0;
    for (let i = 0; i < 50; i++) {
      e.carrying = 0;
      expect(e.step(PICK).reward).toBe(24); // pick never slips
    }
  });

  it('every reset starts on a non-shelf cell, not carrying', () => {
    const e = mk(false, 3);
    const shelves = shelfSet(CLASSIC_LAYOUT);
    for (let i = 0; i < 300; i++) {
      e.reset();
      expect(shelves.has(e.y * 6 + e.x)).toBe(false);
      expect(e.carrying).toBe(0);
    }
    expect(e.validStarts().length).toBe(32);
  });

  it('validates layouts', () => {
    expect(validateLayout(CLASSIC_LAYOUT)).toBeNull();
    expect(validateLayout({ ...CLASSIC_LAYOUT, item: [2, 2] })).not.toBeNull();
    expect(validateLayout({ ...CLASSIC_LAYOUT, dock: CLASSIC_LAYOUT.item })).not.toBeNull();
  });

  it('transition is pure', () => {
    const sh = shelfSet(CLASSIC_LAYOUT);
    const t = transition(CLASSIC_LAYOUT, sh, 0, 4, 0, PICK);
    expect(t.carrying).toBe(1);
    expect(t.reward).toBe(24);
  });
});

describe('optimal planner', () => {
  const plan = solve(CLASSIC_LAYOUT);
  it('matches hand-computed distances', () => {
    // From the item cell: pick (1) + walk to dock (5 right + 1 down = 6) + drop (1) = 8
    expect(plan.dist[(4 * 6 + 0) * 2 + 0]).toBe(8);
    // Carrying at the dock: just drop
    expect(plan.dist[(5 * 6 + 5) * 2 + 1]).toBe(1);
  });
  it('optimal path achieves return 125 - steps in the simulator', () => {
    const e = mk();
    const s = e.reset([5, 0]);
    let total = 0;
    let steps = 0;
    let state = s;
    for (;;) {
      const a = plan.action[state];
      const r = e.step(a);
      total += r.reward;
      steps++;
      state = r.state;
      if (r.terminated) break;
    }
    expect(steps).toBe(plan.dist[(0 * 6 + 5) * 2]);
    expect(total).toBe(125 - steps);
    expect(optimalPath(CLASSIC_LAYOUT, plan, [5, 0]).length).toBeGreaterThan(2);
  });
});
