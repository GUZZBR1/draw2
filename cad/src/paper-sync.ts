import type { Entity } from './geometry.ts';
import { addLayer, LIVE_LAYER, type Doc, type PaperState, type PaperStroke } from './model.ts';

export function strokeToEntity(s: PaperStroke, layer: string, id: string, paperId?: string): Entity {
  const points = s.points.map(([x, y]) => ({ x, y }));
  if (s.kind === 'circle' && s.center && s.radius !== undefined) {
    return {
      id,
      layer,
      kind: 'circle',
      points: [],
      closed: true,
      center: { x: s.center[0], y: s.center[1] },
      radius: s.radius,
      paperId,
    };
  }
  return { id, layer, kind: s.kind === 'line' ? 'line' : 'polyline', points, closed: s.closed, paperId };
}

const paperKey = (session: string, page: number, strokeId: string) => `${session}/${page}/${strokeId}`;

function freezeLive(doc: Doc, label: string): void {
  const live = doc.entities.filter((e) => e.layer === LIVE_LAYER);
  if (!live.length) return;
  const layer = addLayer(doc, label, 'papel-pagina');
  for (const e of live) {
    e.layer = layer.id;
    // Id novo: a camada ao vivo reaproveita p-<página>-<traço> na
    // sessão seguinte e o congelado não pode colidir com ele.
    e.id = `e${doc.nextId++}`;
    delete e.paperId;
  }
}

/**
 * Espelha o estado do papel no documento. Devolve true se mudou algo.
 *
 * Regras: a camada "Papel (ao vivo)" segue o servidor; página virada
 * (ou sessão nova) congela o que estava ao vivo numa camada própria;
 * páginas do histórico ainda não vistas entram congeladas; traço do
 * papel apagado ou editado no CAD fica suprimido e não volta.
 */
export function applyPaperState(doc: Doc, state: PaperState): boolean {
  let changed = false;
  if (state.paper) {
    const { width_mm, height_mm, format } = state.paper;
    if (doc.sheet.width !== width_mm || doc.sheet.height !== height_mm || doc.sheet.format !== format) {
      doc.sheet = { width: width_mm, height: height_mm, format };
      changed = true;
    }
  }
  const before = JSON.stringify(doc.entities);
  if (doc.paper.session !== state.session) {
    if (doc.paper.session !== null) freezeLive(doc, 'Papel (sessão anterior)');
    doc.paper = { session: state.session, page: state.page, importedPages: [], suppressed: [] };
    changed = true;
  } else if (state.page !== doc.paper.page) {
    if (!doc.paper.importedPages.includes(doc.paper.page)) {
      freezeLive(doc, `Papel — página ${doc.paper.page}`);
      doc.paper.importedPages.push(doc.paper.page);
    } else {
      doc.entities = doc.entities.filter((e) => e.layer !== LIVE_LAYER || !e.paperId);
    }
    doc.paper.page = state.page;
    changed = true;
  }
  for (const page of state.history) {
    if (page.page === state.page || doc.paper.importedPages.includes(page.page)) continue;
    const layer = addLayer(doc, `Papel — página ${page.page}`, 'papel-pagina');
    for (const s of page.strokes) {
      if (doc.paper.suppressed.includes(paperKey(state.session, page.page, s.id))) continue;
      doc.entities.push(strokeToEntity(s, layer.id, `e${doc.nextId++}`));
    }
    doc.paper.importedPages.push(page.page);
    changed = true;
  }
  const suppressed = new Set(doc.paper.suppressed);
  const live = new Map(doc.entities.filter((e) => e.layer === LIVE_LAYER && e.paperId).map((e) => [e.paperId!, e]));
  const wanted = new Set<string>();
  const next: Entity[] = doc.entities.filter((e) => !(e.layer === LIVE_LAYER && e.paperId));
  for (const s of state.strokes) {
    const key = paperKey(state.session, state.page, s.id);
    if (suppressed.has(key)) continue;
    wanted.add(key);
    const id = live.get(key)?.id ?? `p-${state.page}-${s.id}`;
    next.push(strokeToEntity(s, LIVE_LAYER, id, key));
  }
  doc.entities = next;
  return changed || JSON.stringify(doc.entities) !== before;
}

/** Traço do papel que o usuário apagou ou editou não volta pelo servidor. */
export function detachFromPaper(doc: Doc, e: Entity): void {
  if (!e.paperId) return;
  if (!doc.paper.suppressed.includes(e.paperId)) doc.paper.suppressed.push(e.paperId);
  delete e.paperId;
}
