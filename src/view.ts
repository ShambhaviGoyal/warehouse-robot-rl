import { TabularAgent } from './agents';
import { Cell, Dir, Layout, encode, shelfSet } from './env';
import { cssVar, sizeCanvas } from './charts';

const SPRITE_NAMES = [
  'warehouse_floor',
  'dock',
  'item',
  'shelf',
  'robot_collision',
  ...(['up', 'down', 'left', 'right'] as const).flatMap((d) => [`robot_${d}`, `robot_carrying_${d}`]),
];

const sprites: Record<string, HTMLImageElement> = {};

export function loadSprites(onLoad: () => void): void {
  for (const name of SPRITE_NAMES) {
    const img = new Image();
    img.onload = onLoad;
    img.src = `${import.meta.env.BASE_URL}sprites/${name}.png`;
    sprites[name] = img;
  }
}

const ready = (name: string): HTMLImageElement | null => {
  const img = sprites[name];
  return img && img.complete && img.naturalWidth > 0 ? img : null;
};

export interface RobotPose {
  x: number;
  y: number;
  carrying: number;
  dir: Dir;
  collided: boolean;
  delivered: boolean;
}

export interface ViewState {
  layout: Layout;
  agent: TabularAgent | null;
  /** 0 = searching for the item, 1 = carrying it. Controls which half of the Q-table is drawn. */
  phase: 0 | 1;
  showHeat: boolean;
  showPolicy: boolean;
  showOptimal: boolean;
  optimalPath: Cell[];
  trail: Cell[];
  robot: RobotPose;
  start: Cell;
  hover: Cell | null;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function hexRgb(h: string): number[] {
  const m = /^#?([0-9a-f]{6})$/i.exec(h.trim());
  const n = m ? parseInt(m[1], 16) : 0x808080;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Three-stop ramp whose colours come from the active theme. */
function ramp(t: number, stops: number[][]): string {
  const k = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(k));
  const f = k - i;
  const c = stops[i].map((v, j) => Math.round(lerp(v, stops[i + 1][j], f)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function cellFromEvent(canvas: HTMLCanvasElement, e: MouseEvent | PointerEvent, size: number): Cell | null {
  const r = canvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - r.left) / r.width) * size);
  const y = Math.floor(((e.clientY - r.top) / r.height) * size);
  return x >= 0 && y >= 0 && x < size && y < size ? [x, y] : null;
}

function drawArrow(ctx: CanvasRenderingContext2D, cx: number, cy: number, dx: number, dy: number, len: number): void {
  const x1 = cx - (dx * len) / 2;
  const y1 = cy - (dy * len) / 2;
  const x2 = cx + (dx * len) / 2;
  const y2 = cy + (dy * len) / 2;
  const head = len * 0.38;
  const ang = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(ang - 0.55), y2 - head * Math.sin(ang - 0.55));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(ang + 0.55), y2 - head * Math.sin(ang + 0.55));
  ctx.stroke();
}

export function renderGrid(canvas: HTMLCanvasElement, v: ViewState): void {
  const { w, ctx } = sizeCanvas(canvas);
  const n = v.layout.size;
  const cell = w / n;
  const shelves = shelfSet(v.layout);
  ctx.clearRect(0, 0, w, w);

  // Floor
  const floor = ready('warehouse_floor');
  if (floor) ctx.drawImage(floor, 0, 0, w, w);
  else {
    ctx.fillStyle = '#1b2440';
    ctx.fillRect(0, 0, w, w);
  }
  ctx.fillStyle = cssVar('--floor-tint', 'rgba(8,12,28,0.35)');
  ctx.fillRect(0, 0, w, w);

  // Value heatmap (max Q for the chosen phase)
  const agent = v.agent;
  if (agent && v.showHeat) {
    const stops = [cssVar('--heat-0', '#1e295a'), cssVar('--heat-1', '#14a0aa'), cssVar('--heat-2', '#fac83c')].map(hexRgb);
    const heatAlpha = parseFloat(cssVar('--heat-alpha', '0.62')) || 0.62;
    const vals: (number | null)[] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        if (shelves.has(y * n + x)) {
          vals.push(null);
          continue;
        }
        const s = encode(n, x, y, v.phase);
        if (agent.untouched(s)) {
          vals.push(null);
          continue;
        }
        const q = agent.maxValue(s);
        vals.push(q);
        lo = Math.min(lo, q);
        hi = Math.max(hi, q);
      }
    vals.forEach((q, i) => {
      if (q === null) return;
      const x = i % n;
      const y = Math.floor(i / n);
      ctx.fillStyle = ramp(hi > lo ? (q - lo) / (hi - lo) : 0.5, stops);
      ctx.globalAlpha = heatAlpha;
      ctx.fillRect(x * cell, y * cell, cell, cell);
      ctx.globalAlpha = 1;
    });
  }

  // Grid lines
  ctx.strokeStyle = cssVar('--cell-line', 'rgba(255,255,255,0.14)');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    ctx.moveTo(Math.round(i * cell) + 0.5, 0);
    ctx.lineTo(Math.round(i * cell) + 0.5, w);
    ctx.moveTo(0, Math.round(i * cell) + 0.5);
    ctx.lineTo(w, Math.round(i * cell) + 0.5);
  }
  ctx.stroke();

  // Start marker
  ctx.strokeStyle = cssVar('--start', 'rgba(56,189,248,0.9)');
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(v.start[0] * cell + 3, v.start[1] * cell + 3, cell - 6, cell - 6);
  ctx.setLineDash([]);

  const drawSprite = (name: string, x: number, y: number, scale = 0.86, alpha = 1) => {
    const img = ready(name);
    if (!img) return;
    const size = cell * scale;
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, x * cell + (cell - size) / 2, y * cell + (cell - size) / 2, size, size);
    ctx.globalAlpha = 1;
  };

  // Optimal path (dashed cyan)
  const line = (pts: Cell[], color: string, width: number, dash: number[]) => {
    if (pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const px = (x + 0.5) * cell;
      const py = (y + 0.5) * cell;
      if (i) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };
  if (v.showOptimal) line(v.optimalPath, cssVar('--path', 'rgba(255,255,255,0.85)'), Math.max(2, cell * 0.06), [6, 6]);

  // Sprites
  drawSprite('dock', v.layout.dock[0], v.layout.dock[1]);
  if (v.robot.carrying === 0 && !v.robot.delivered) drawSprite('item', v.layout.item[0], v.layout.item[1]);
  for (const [x, y] of v.layout.shelves) drawSprite('shelf', x, y);

  // Policy arrows
  if (agent && v.showPolicy) {
    ctx.lineWidth = Math.max(2, cell * 0.055);
    ctx.lineCap = 'round';
    ctx.strokeStyle = cssVar('--arrow', 'rgba(255,255,255,0.95)');
    ctx.fillStyle = cssVar('--arrow', 'rgba(255,255,255,0.95)');
    ctx.shadowColor = cssVar('--arrow-shadow', 'rgba(0,0,0,0.65)');
    ctx.shadowBlur = 4;
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        if (shelves.has(y * n + x)) continue;
        const s = encode(n, x, y, v.phase);
        if (agent.untouched(s)) continue;
        const a = agent.greedy(s);
        const cx = (x + 0.5) * cell;
        const cy = (y + 0.5) * cell;
        if (a < 4) {
          const d = [
            [0, -1],
            [0, 1],
            [-1, 0],
            [1, 0],
          ][a];
          drawArrow(ctx, cx, cy, d[0], d[1], cell * 0.42);
        } else {
          ctx.font = `700 ${Math.max(9, cell * 0.2)}px ${cssVar('--font-body', 'ui-sans-serif, system-ui, sans-serif')}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(a === 4 ? 'PICK' : 'DROP', cx, cy + cell * 0.3);
        }
      }
    ctx.shadowBlur = 0;
  }

  // Replay trail
  if (v.trail.length > 1) line(v.trail, cssVar('--trail', 'rgba(251,191,36,0.9)'), Math.max(2.5, cell * 0.07), []);

  // Hover outline
  if (v.hover) {
    ctx.strokeStyle = cssVar('--hover', 'rgba(255,255,255,0.8)');
    ctx.lineWidth = 2;
    ctx.strokeRect(v.hover[0] * cell + 1, v.hover[1] * cell + 1, cell - 2, cell - 2);
  }

  // Robot
  const r = v.robot;
  const file = r.collided ? 'robot_collision' : r.carrying ? `robot_carrying_${r.dir}` : `robot_${r.dir}`;
  drawSprite(file, r.x, r.y, 0.8);
}
