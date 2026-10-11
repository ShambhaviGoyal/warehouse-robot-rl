import { describe, expect, it } from 'vitest';
import { TabularAgent } from '../src/agents';
import { WarehouseEnv, mulberry32, validateLayout } from '../src/env';
import { evaluate } from '../src/evaluate';
import { PRESETS } from '../src/presets';
import { isSolvable, solve } from '../src/planner';

describe('presets', () => {
  for (const p of PRESETS) {
    it(`${p.id}: valid, solvable, and Q-learning learns it`, () => {
      expect(validateLayout(p.layout)).toBeNull();
      const plan = solve(p.layout);
      expect(isSolvable(p.layout, plan)).toBe(true);
      const rng = mulberry32(1);
      const env = new WarehouseEnv(p.layout, false, rng, p.maxSteps);
      const agent = new TabularAgent(env.nStates, 'q', p.hyper, rng);
      for (let i = 0; i < p.episodes; i++) agent.trainEpisode(env);
      const ev = evaluate(agent, p.layout, plan, false, p.maxSteps, 0.1);
      expect(ev.successRate).toBeGreaterThan(0.95);
    }, 60000);
  }
});
