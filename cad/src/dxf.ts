import { circlePoints, type Entity } from './geometry.ts';
import type { Doc } from './model.ts';

const ACI: [number, number, number, number][] = [
  [1, 255, 0, 0], [2, 255, 255, 0], [3, 0, 255, 0], [4, 0, 255, 255],
  [5, 0, 0, 255], [6, 255, 0, 255], [7, 0, 0, 0], [8, 128, 128, 128],
  [30, 255, 127, 0], [94, 0, 127, 0], [140, 0, 191, 255], [150, 0, 127, 255],
  [174, 0, 127, 127], [214, 127, 0, 127],
];

/** Cor hex -> índice ACI mais próximo da paleta básica do AutoCAD. */
export function hexToAci(hex: string): number {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return 7;
  const v = parseInt(m[1], 16);
  const r = (v >> 16) & 255, g = (v >> 8) & 255, b = v & 255;
  let best = 7, bestD = Infinity;
  for (const [i, cr, cg, cb] of ACI) {
    const d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

const sanitizeLayer = (name: string) => name.replace(/[<>/\\":;?*|=`]/g, '_').slice(0, 255) || '0';
const n = (v: number) => (Math.round(v * 1000) / 1000).toFixed(3);

/**
 * DXF R12 (AC1009) ASCII em mm, Y para cima (y_cad = altura - y).
 * R12 não exige handles e abre no AutoCAD, LibreCAD e afins.
 */
export function toDxf(doc: Doc): string {
  const h = doc.sheet.height;
  const visible = doc.entities.filter((e) => doc.layers.find((l) => l.id === e.layer)?.visible !== false);
  const names = new Map(doc.layers.map((l) => [l.id, sanitizeLayer(l.name)]));
  const out: string[] = [
    '0', 'SECTION', '2', 'HEADER',
    '9', '$ACADVER', '1', 'AC1009',
    '9', '$INSUNITS', '70', '4',
    '9', '$EXTMIN', '10', '0.0', '20', '0.0', '30', '0.0',
    '9', '$EXTMAX', '10', n(doc.sheet.width), '20', n(h), '30', '0.0',
    '0', 'ENDSEC',
    '0', 'SECTION', '2', 'TABLES',
    '0', 'TABLE', '2', 'LAYER', '70', String(doc.layers.length),
  ];
  for (const l of doc.layers) {
    out.push('0', 'LAYER', '2', names.get(l.id)!, '70', '0', '62', String(hexToAci(l.color)), '6', 'CONTINUOUS');
  }
  out.push('0', 'ENDTAB', '0', 'ENDSEC', '0', 'SECTION', '2', 'ENTITIES');
  const y = (v: number) => n(h - v);
  const writeEntity = (e: Entity) => {
    const layer = names.get(e.layer) ?? '0';
    if (e.kind === 'circle' && e.center && e.radius !== undefined) {
      out.push('0', 'CIRCLE', '8', layer, '10', n(e.center.x), '20', y(e.center.y), '30', '0.0', '40', n(e.radius));
      return;
    }
    const pts = e.kind === 'circle' && e.center && e.radius ? circlePoints(e.center, e.radius) : e.points;
    if (pts.length < 2) return;
    if (e.kind === 'line' && pts.length === 2) {
      out.push('0', 'LINE', '8', layer, '10', n(pts[0].x), '20', y(pts[0].y), '30', '0.0', '11', n(pts[1].x), '21', y(pts[1].y), '31', '0.0');
      return;
    }
    out.push('0', 'POLYLINE', '8', layer, '66', '1', '10', '0.0', '20', '0.0', '30', '0.0', '70', e.closed ? '1' : '0');
    for (const p of pts) out.push('0', 'VERTEX', '8', layer, '10', n(p.x), '20', y(p.y), '30', '0.0');
    out.push('0', 'SEQEND', '8', layer);
  };
  visible.forEach(writeEntity);
  out.push('0', 'ENDSEC', '0', 'EOF');
  return out.join('\n') + '\n';
}
