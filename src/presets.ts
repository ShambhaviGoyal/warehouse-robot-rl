import { Layout, CLASSIC_LAYOUT, Cell } from './env';
import { Hyper, DEFAULT_HYPER } from './agents';

export interface Preset {
  id: string;
  name: string;
  blurb: string;
  layout: Layout;
  maxSteps: number;
  episodes: number;
  hyper: Hyper;
}

const column = (x: number, from: number, to: number): Cell[] =>
  Array.from({ length: to - from + 1 }, (_, i) => [x, from + i] as Cell);

export const PRESETS: Preset[] = [
  {
    id: 'classic',
    name: 'Classic 6×6',
    blurb: 'The original assignment map: one shelf block, one item, one dock.',
    layout: CLASSIC_LAYOUT,
    maxSteps: 100,
    episodes: 1500,
    hyper: DEFAULT_HYPER,
  },
  {
    id: 'open',
    name: 'Open floor',
    blurb: 'No shelves. The easiest map, useful as a sanity check.',
    layout: { size: 6, shelves: [], item: [0, 4], dock: [5, 5] },
    maxSteps: 100,
    episodes: 1000,
    hyper: DEFAULT_HYPER,
  },
  {
    id: 'aisles',
    name: 'Aisles 8×8',
    blurb: 'Three rows of shelf pairs with long aisles between them.',
    layout: {
      size: 8,
      shelves: [
        [1, 1], [2, 1], [4, 1], [5, 1],
        [1, 3], [2, 3], [4, 3], [5, 3],
        [1, 5], [2, 5], [4, 5], [5, 5],
      ],
      item: [0, 7],
      dock: [7, 7],
    },
    maxSteps: 150,
    episodes: 2500,
    hyper: DEFAULT_HYPER,
  },
  {
    id: 'maze',
    name: 'Serpentine maze 8×8',
    blurb: 'A winding route with only one way through. Hard: sparse reward, long horizon.',
    layout: {
      size: 8,
      shelves: [...column(1, 0, 5), ...column(3, 2, 7), ...column(5, 0, 5)],
      item: [0, 0],
      dock: [7, 0],
    },
    maxSteps: 200,
    episodes: 6000,
    hyper: { ...DEFAULT_HYPER, epsDecay: 0.999, gamma: 0.98 },
  },
];

export const presetById = (id: string): Preset => PRESETS.find((p) => p.id === id) ?? PRESETS[0];
