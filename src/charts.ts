export interface Series {
  label: string;
  color: string;
  data: number[];
  /** Optional x positions (same length as data). Defaults to the index. */
  xs?: number[];
  band?: { lo: number[]; hi: number[] };
  dashed?: boolean;
  width?: number;
}

export interface ChartOptions {
  series: Series[];
  yMin?: number;
  yMax?: number;
  xMax?: number;
  yFormat?: (v: number) => string;
  xLabel?: string;
  ref?: { y: number; label: string; color: string };
}

export const cssVar = (name: string, fallback: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function niceTicks(lo: number, hi: number, count: number): number[] {
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < 1e-9 ? 0 : v);
  return out;
}

export function sizeCanvas(canvas: HTMLCanvasElement): { w: number; h: number; ctx: CanvasRenderingContext2D } {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx };
}

export function drawChart(canvas: HTMLCanvasElement, opt: ChartOptions): void {
  const { w, h, ctx } = sizeCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (w < 40 || h < 40) return;

  const grid = cssVar('--grid', 'rgba(255,255,255,0.08)');
  const text = cssVar('--muted', '#8b97b5');
  const pad = { l: 46, r: 20, t: 10, b: opt.xLabel ? 36 : 20 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;

  let xMax = opt.xMax ?? 1;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of opt.series) {
    s.data.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
      xMax = Math.max(xMax, s.xs ? s.xs[i] : i + 1);
    });
    s.band?.lo.forEach((v) => Number.isFinite(v) && (lo = Math.min(lo, v)));
    s.band?.hi.forEach((v) => Number.isFinite(v) && (hi = Math.max(hi, v)));
  }
  if (opt.ref) {
    lo = Math.min(lo, opt.ref.y);
    hi = Math.max(hi, opt.ref.y);
  }
  if (!Number.isFinite(lo)) {
    lo = 0;
    hi = 1;
  }
  const yMin = opt.yMin ?? lo - (hi - lo) * 0.05;
  let yMax = opt.yMax ?? hi + (hi - lo) * 0.08;
  if (yMax <= yMin) yMax = yMin + 1;

  const X = (x: number) => pad.l + (x / xMax) * pw;
  const Y = (y: number) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * ph;
  const fmt = opt.yFormat ?? ((v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1)));

  ctx.font = `11px ${cssVar('--font-body', 'ui-sans-serif, system-ui, sans-serif')}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  ctx.lineWidth = 1;
  for (const t of niceTicks(yMin, yMax, 4)) {
    const y = Math.round(Y(t)) + 0.5;
    ctx.strokeStyle = grid;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillStyle = text;
    ctx.fillText(fmt(t), pad.l - 6, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of niceTicks(0, xMax, 5)) {
    ctx.fillStyle = text;
    ctx.fillText(String(Math.round(t)), X(t), h - pad.b + 5);
  }
  if (opt.xLabel) {
    ctx.fillStyle = text;
    ctx.fillText(opt.xLabel, pad.l + pw / 2, h - 14);
  }

  if (opt.ref) {
    ctx.strokeStyle = opt.ref.color;
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(opt.ref.y));
    ctx.lineTo(w - pad.r, Y(opt.ref.y));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = opt.ref.color;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(opt.ref.label, w - pad.r - 2, Y(opt.ref.y) - 2);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.l, pad.t - 2, pw, ph + 4);
  ctx.clip();
  for (const s of opt.series) {
    const px = (i: number) => X(s.xs ? s.xs[i] : i + 1);
    if (s.band) {
      ctx.fillStyle = s.color + '2e';
      ctx.beginPath();
      s.data.forEach((_, i) => (i ? ctx.lineTo(px(i), Y(s.band!.hi[i])) : ctx.moveTo(px(i), Y(s.band!.hi[i]))));
      for (let i = s.data.length - 1; i >= 0; i--) ctx.lineTo(px(i), Y(s.band.lo[i]));
      ctx.closePath();
      ctx.fill();
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width ?? 2;
    ctx.lineJoin = 'round';
    ctx.setLineDash(s.dashed ? [5, 4] : []);
    ctx.beginPath();
    let started = false;
    s.data.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      if (started) ctx.lineTo(px(i), Y(v));
      else {
        ctx.moveTo(px(i), Y(v));
        started = true;
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

/** Keep at most `max` evenly spaced points so long runs stay cheap to draw. */
export function downsample<T>(xs: T[], max: number): { values: T[]; index: number[] } {
  if (xs.length <= max) return { values: xs, index: xs.map((_, i) => i) };
  const values: T[] = [];
  const index: number[] = [];
  const step = (xs.length - 1) / (max - 1);
  for (let i = 0; i < max; i++) {
    const j = Math.round(i * step);
    values.push(xs[j]);
    index.push(j);
  }
  return { values, index };
}
