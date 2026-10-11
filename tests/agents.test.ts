import { describe, expect, it } from 'vitest';
import { Algo, DEFAULT_HYPER, TabularAgent } from '../src/agents';
import { CLASSIC_LAYOUT, WarehouseEnv, mulberry32 } from '../src/env';
import { evaluate } from '../src/evaluate';
import { solve } from '../src/planner';

function train(algo: Algo, stochastic: boolean, episodes = 1000, seed = 1) {
  const rng = mulberry32(seed);
  const env = new WarehouseEnv(CLASSIC_LAYOUT, stochastic, rng);
  const agent = new TabularAgent(env.nStates, algo, DEFAULT_HYPER, rng);
  for (let i = 0; i < episodes; i++) agent.trainEpisode(env);
  return { agent, env };
}

describe('agents learn the classic warehouse', () => {
  const plan = solve(CLASSIC_LAYOUT);
  for (const algo of ['q', 'sarsa', 'expected', 'double'] as Algo[]) {
    it(`${algo} (deterministic) delivers from every start`, () => {
      const { agent } = train(algo, false, 1500);
      const ev = evaluate(agent, CLASSIC_LAYOUT, plan, false, 100, 0.1);
      expect(ev.successRate).toBe(1);
      expect(ev.efficiency).toBeGreaterThan(0.8);
    });
  }
  it('q-learning (stochastic) still succeeds', () => {
    const { agent } = train('q', true, 1500);
    const ev = evaluate(agent, CLASSIC_LAYOUT, plan, true, 100, 0.1);
    expect(ev.successRate).toBeGreaterThan(0.9);
  });
  it('is reproducible for a fixed seed', () => {
    const a = train('q', false, 200, 5).agent.q;
    const b = train('q', false, 200, 5).agent.q;
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});
