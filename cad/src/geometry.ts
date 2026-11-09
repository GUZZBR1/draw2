export type Pt = { x: number; y: number };

export type EntityKind = 'line' | 'polyline' | 'circle';

export type Entity = {
  id: string;
  layer: string;
  kind: EntityKind;
  points: Pt[];
  closed: boolean;
  center?: Pt;
  radius?: number;
  /** Traço vindo do papel ao vivo (id do servidor); some ao editar. */
  paperId?: string;
};

export type Box = { minX: number; minY: number; maxX: number; maxY: number };

export const dist = (a: Pt, b: Pt): number => Math.hypot(a.x - b.x, a.y - b.y);

export function distToSegment(p: Pt, a: Pt, b: Pt): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return dist(p, a);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2));
  return dist(p, { x: a.x + t * dx, y: a.y + t * dy });
}

export function segments(e: Entity): [Pt, Pt][] {
  const pts = e.points;
  const out: [Pt, Pt][] = [];
  for (let i = 0; i + 1 < pts.length; i++) out.push([pts[i], pts[i + 1]]);
  if (e.closed && pts.length > 2) out.push([pts[pts.length - 1], pts[0]]);
  return out;
}

export function distToEntity(p: Pt, e: Entity): number {
  if (e.kind === 'circle' && e.center && e.radius !== undefined) {
    return Math.abs(dist(p, e.center) - e.radius);
  }
  if (e.points.length === 1) return dist(p, e.points[0]);
  let best = Infinity;
  for (const [a, b] of segments(e)) best = Math.min(best, distToSegment(p, a, b));
  return best;
}

export function entityBox(e: Entity): Box {
  if (e.kind === 'circle' && e.center && e.radius !== undefined) {
    const { x, y } = e.center;
    const r = e.radius;
    return { minX: x - r, minY: y - r, maxX: x + r, maxY: y + r };
  }
  const xs = e.points.map((p) => p.x);
  const ys = e.points.map((p) => p.y);
  return { minX: Math.min(...xs), minY: Math.min(...ys), maxX: Math.max(...xs), maxY: Math.max(...ys) };
}

export function unionBox(boxes: Box[]): Box | null {
  if (!boxes.length) return null;
  return boxes.reduce((a, b) => ({
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  }));
}

export function boxContains(outer: Box, inner: Box): boolean {
  return inner.minX >= outer.minX && inner.maxX <= outer.maxX && inner.minY >= outer.minY && inner.maxY <= outer.maxY;
}

export function normalizeBox(a: Pt, b: Pt): Box {
  return { minX: Math.min(a.x, b.x), minY: Math.min(a.y, b.y), maxX: Math.max(a.x, b.x), maxY: Math.max(a.y, b.y) };
}

export function translate(e: Entity, dx: number, dy: number): Entity {
  return {
    ...e,
    points: e.points.map((p) => ({ x: p.x + dx, y: p.y + dy })),
    center: e.center ? { x: e.center.x + dx, y: e.center.y + dy } : undefined,
  };
}

export function circlePoints(c: Pt, r: number, n = 64): Pt[] {
  return Array.from({ length: n }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    return { x: c.x + r * Math.cos(t), y: c.y + r * Math.sin(t) };
  });
}

export function rectPoints(a: Pt, b: Pt): Pt[] {
  return [
    { x: a.x, y: a.y },
    { x: b.x, y: a.y },
    { x: b.x, y: b.y },
    { x: a.x, y: b.y },
  ];
}

export function entityLength(e: Entity): number {
  if (e.kind === 'circle' && e.radius !== undefined) return 2 * Math.PI * e.radius;
  return segments(e).reduce((s, [a, b]) => s + dist(a, b), 0);
}

/** Pontos de snap: extremidades, vértices, pontos médios e centros. */
export function snapPoints(e: Entity): Pt[] {
  if (e.kind === 'circle' && e.center && e.radius !== undefined) {
    const { x, y } = e.center;
    const r = e.radius;
    return [e.center, { x: x + r, y }, { x: x - r, y }, { x, y: y + r }, { x, y: y - r }];
  }
  const mids = segments(e).map(([a, b]) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }));
  return [...e.points, ...mids];
}

export function snapToGrid(p: Pt, step: number): Pt {
  return { x: Math.round(p.x / step) * step, y: Math.round(p.y / step) * step };
}
