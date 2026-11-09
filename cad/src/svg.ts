import type { Doc } from './model.ts';

const n = (v: number) => String(Math.round(v * 1000) / 1000);
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

/** SVG em mm, mesmo sistema do papel (origem no topo, y para baixo). */
export function toSvg(doc: Doc): string {
  const { width, height } = doc.sheet;
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${n(width)}mm" height="${n(height)}mm" viewBox="0 0 ${n(width)} ${n(height)}">`,
  ];
  for (const layer of doc.layers) {
    if (!layer.visible) continue;
    const items = doc.entities.filter((e) => e.layer === layer.id);
    if (!items.length) continue;
    parts.push(`<g id="${esc(layer.id)}" inkscape:label="${esc(layer.name)}" fill="none" stroke="${esc(layer.color)}" stroke-width="0.35" stroke-linecap="round" stroke-linejoin="round" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">`);
    for (const e of items) {
      if (e.kind === 'circle' && e.center && e.radius !== undefined) {
        parts.push(`<circle cx="${n(e.center.x)}" cy="${n(e.center.y)}" r="${n(e.radius)}"/>`);
      } else if (e.points.length >= 2) {
        const pts = e.points.map((p) => `${n(p.x)},${n(p.y)}`).join(' ');
        parts.push(e.closed ? `<polygon points="${pts}"/>` : `<polyline points="${pts}"/>`);
      }
    }
    parts.push('</g>');
  }
  parts.push('</svg>');
  return parts.join('\n') + '\n';
}
