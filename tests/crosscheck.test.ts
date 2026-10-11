import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { CLASSIC_LAYOUT, WarehouseEnv } from '../src/env';

// original_trace.json is recorded from the original Python environment by
// scripts/crosscheck_original.py. Replaying it here proves the port has identical rules.
const traces = JSON.parse(readFileSync(new URL('./original_trace.json', import.meta.url), 'utf8')) as {
  start: [number, number];
  actions: number[];
  steps: [number, number, number, number, boolean, boolean][];
}[];

describe('TypeScript port matches the original Python environment', () => {
  traces.forEach((t, i) => {
    it(`trace ${i} from ${JSON.stringify(t.start)} (${t.steps.length} steps)`, () => {
      const env = new WarehouseEnv(CLASSIC_LAYOUT, false, Math.random, 100);
      env.reset(t.start);
      t.actions.forEach((a, k) => {
        const r = env.step(a);
        const [x, y, c, reward, term, trunc] = t.steps[k];
        expect([env.x, env.y, env.carrying]).toEqual([x, y, c]);
        expect(r.reward).toBe(reward);
        expect(r.terminated).toBe(term);
        expect(r.truncated).toBe(trunc);
      });
    });
  });
  it('includes at least one successful delivery', () => {
    expect(traces.some((t) => t.steps.some((s) => s[4] === true))).toBe(true);
  });
});
