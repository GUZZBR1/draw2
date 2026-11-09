import './style.css';
import {
  boxContains,
  dist,
  distToEntity,
  entityBox,
  entityLength,
  normalizeBox,
  rectPoints,
  snapPoints,
  snapToGrid,
  translate,
  unionBox,
  type Entity,
  type Pt,
} from './geometry.ts';
import { addLayer, cloneDoc, DRAW_LAYER, layerOf, LIVE_LAYER, newDoc, newId, parseDoc, type Doc, type PaperState } from './model.ts';
import { applyPaperState, detachFromPaper } from './paper-sync.ts';
import { toDxf } from './dxf.ts';
import { toSvg } from './svg.ts';
import { displayColor, nextTheme, parseTheme, THEME_KEY, THEME_LABEL, type ThemeChoice } from './theme.ts';

type Tool = 'select' | 'line' | 'polyline' | 'rect' | 'circle' | 'pan';
type Drag =
  | { mode: 'pan'; sx: number; sy: number; ox: number; oy: number }
  | { mode: 'move'; start: Pt; originals: Entity[]; moved: boolean }
  | { mode: 'box'; start: Pt; end: Pt; additive: boolean };

const STORAGE_KEY = 'draw-cad-doc-v1';
const HIT_PX = 6;
const SNAP_PX = 9;
const TOOLS: { id: Tool; label: string; key: string; hint: string }[] = [
  { id: 'select', label: 'Selecionar', key: 'V', hint: 'Clique ou arraste para selecionar; arraste a seleção para mover.' },
  { id: 'line', label: 'Linha', key: 'L', hint: 'Clique o início e o fim. Continua do último ponto; Esc termina. Shift: ortogonal.' },
  { id: 'polyline', label: 'Polilinha', key: 'P', hint: 'Clique os vértices; Enter ou duplo clique termina; clique no 1º ponto fecha.' },
  { id: 'rect', label: 'Retângulo', key: 'R', hint: 'Clique dois cantos opostos.' },
  { id: 'circle', label: 'Círculo', key: 'C', hint: 'Clique o centro e depois um ponto do raio.' },
  { id: 'pan', label: 'Mover vista', key: 'H', hint: 'Arraste para mover a vista. Roda do mouse: zoom.' },
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('view');
const ctx = canvas.getContext('2d')!;

let doc = loadDoc();
let undoStack: Doc[] = [];
let redoStack: Doc[] = [];
let lastPaper: PaperState | null = null;
let selection = new Set<string>();
let tool: Tool = 'select';
let view = { scale: 2, ox: 0, oy: 0 };
let cursor: Pt = { x: 0, y: 0 };
let snapped: { p: Pt; onEntity: boolean } | null = null;
let draft: Pt[] = [];
let drag: Drag | null = null;
let hover: string | null = null;
let spaceDown = false;
let theme: ThemeChoice = parseTheme(safeGet(THEME_KEY));
const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');

function safeGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

const isDark = () => theme === 'dark' || (theme === 'system' && darkQuery.matches);
const cssVar = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
const layerColor = (id: string) => displayColor(layerOf(doc, id)?.color ?? '#000000', isDark());

function applyTheme(): void {
  if (theme === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  const button = document.getElementById('theme');
  if (button) button.textContent = `Tema: ${THEME_LABEL[theme]}`;
  render();
}

function loadDoc(): Doc {
  try {
    const text = localStorage.getItem(STORAGE_KEY);
    if (text) return parseDoc(text);
  } catch {
    /* documento salvo inválido: começa um novo */
  }
  return newDoc();
}

function save(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(doc));
  } catch {
    /* sem armazenamento: o desenho continua na memória */
  }
}

function commit(mutate: (d: Doc) => void): void {
  undoStack.push(cloneDoc(doc));
  if (undoStack.length > 200) undoStack.shift();
  redoStack = [];
  mutate(doc);
  afterChange();
}

function afterChange(): void {
  selection = new Set([...selection].filter((id) => doc.entities.some((e) => e.id === id && isEditable(e))));
  save();
  refreshPanels();
  render();
}

function restore(from: Doc[], to: Doc[]): void {
  const previous = from.pop();
  if (!previous) return;
  to.push(cloneDoc(doc));
  doc = previous;
  if (lastPaper) applyPaperState(doc, lastPaper);
  afterChange();
}

const undo = () => restore(undoStack, redoStack);
const redo = () => restore(redoStack, undoStack);

// ---------------------------------------------------------------- vista

const toWorld = (sx: number, sy: number): Pt => ({ x: (sx - view.ox) / view.scale, y: (sy - view.oy) / view.scale });
const toScreen = (p: Pt) => ({ x: p.x * view.scale + view.ox, y: p.y * view.scale + view.oy });

function resize(): void {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}

function fit(): void {
  const rect = canvas.getBoundingClientRect();
  const visible = doc.entities.filter(isVisible).map(entityBox);
  const box = unionBox([{ minX: 0, minY: 0, maxX: doc.sheet.width, maxY: doc.sheet.height }, ...visible])!;
  const margin = 24;
  const scale = Math.min((rect.width - 2 * margin) / (box.maxX - box.minX), (rect.height - 2 * margin) / (box.maxY - box.minY));
  view.scale = Math.max(0.2, Math.min(scale, 80));
  view.ox = (rect.width - (box.maxX - box.minX) * view.scale) / 2 - box.minX * view.scale;
  view.oy = (rect.height - (box.maxY - box.minY) * view.scale) / 2 - box.minY * view.scale;
  render();
}

function zoomAt(sx: number, sy: number, factor: number): void {
  const before = toWorld(sx, sy);
  view.scale = Math.max(0.2, Math.min(view.scale * factor, 80));
  view.ox = sx - before.x * view.scale;
  view.oy = sy - before.y * view.scale;
  render();
}

// ------------------------------------------------------------ consultas

function isVisible(e: Entity): boolean {
  return layerOf(doc, e.layer)?.visible !== false;
}

function isEditable(e: Entity): boolean {
  const layer = layerOf(doc, e.layer);
  return layer?.visible !== false && layer?.locked !== true;
}

function hitTest(p: Pt): Entity | null {
  const tol = HIT_PX / view.scale;
  let best: Entity | null = null;
  let bestD = tol;
  for (const e of doc.entities) {
    if (!isEditable(e)) continue;
    const d = distToEntity(p, e);
    if (d <= bestD) {
      bestD = d;
      best = e;
    }
  }
  return best;
}

function snap(p: Pt, free: boolean): { p: Pt; onEntity: boolean } {
  const snapOn = $<HTMLInputElement>('snap').checked && !free;
  if (!snapOn) return { p, onEntity: false };
  const tol = SNAP_PX / view.scale;
  let best: Pt | null = null;
  let bestD = tol;
  for (const e of doc.entities) {
    if (!isVisible(e)) continue;
    for (const q of snapPoints(e)) {
      const d = dist(p, q);
      if (d < bestD) {
        bestD = d;
        best = q;
      }
    }
  }
  for (const q of draft) {
    const d = dist(p, q);
    if (d < bestD) {
      bestD = d;
      best = q;
    }
  }
  if (best) return { p: { ...best }, onEntity: true };
  return { p: snapToGrid(p, view.scale >= 3 ? 1 : 5), onEntity: false };
}

function ortho(p: Pt, shift: boolean): Pt {
  const last = draft[draft.length - 1];
  if (!shift || !last) return p;
  return Math.abs(p.x - last.x) >= Math.abs(p.y - last.y) ? { x: p.x, y: last.y } : { x: last.x, y: p.y };
}

// ------------------------------------------------------------- desenho

function drawEntity(e: Entity, color: string, width: number, dashed = false): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.beginPath();
  if (e.kind === 'circle' && e.center && e.radius !== undefined) {
    const c = toScreen(e.center);
    ctx.arc(c.x, c.y, e.radius * view.scale, 0, Math.PI * 2);
  } else {
    e.points.forEach((p, i) => {
      const s = toScreen(p);
      if (i === 0) ctx.moveTo(s.x, s.y);
      else ctx.lineTo(s.x, s.y);
    });
    if (e.closed) ctx.closePath();
    if (e.points.length === 1) {
      const s = toScreen(e.points[0]);
      ctx.arc(s.x, s.y, 1.5, 0, Math.PI * 2);
    }
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function previewEntity(): Entity | null {
  const p = snapped?.p ?? cursor;
  const base = { id: 'preview', layer: doc.activeLayer, closed: false } as const;
  if (!draft.length) return null;
  switch (tool) {
    case 'line':
      return { ...base, kind: 'line', points: [draft[0], p] };
    case 'polyline':
      return { ...base, kind: 'polyline', points: [...draft, p] };
    case 'rect':
      return { ...base, kind: 'polyline', points: rectPoints(draft[0], p), closed: true };
    case 'circle':
      return { ...base, kind: 'circle', points: [], center: draft[0], radius: dist(draft[0], p), closed: true };
    default:
      return null;
  }
}

function render(): void {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.fillStyle = cssVar('--canvas-bg', '#dee2e6');
  ctx.fillRect(0, 0, w, h);

  const a = toScreen({ x: 0, y: 0 });
  const b = toScreen({ x: doc.sheet.width, y: doc.sheet.height });
  ctx.fillStyle = cssVar('--sheet', '#ffffff');
  ctx.shadowColor = cssVar('--sheet-shadow', 'rgba(0,0,0,0.18)');
  ctx.shadowBlur = 12;
  ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
  ctx.shadowBlur = 0;

  for (const step of [5, 10]) {
    if (step * view.scale < 7) continue;
    ctx.strokeStyle = step === 10 ? cssVar('--grid-major', '#e3e8ee') : cssVar('--grid-minor', '#f1f3f5');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = step; x < doc.sheet.width; x += step) {
      const s = toScreen({ x, y: 0 });
      ctx.moveTo(Math.round(s.x) + 0.5, a.y);
      ctx.lineTo(Math.round(s.x) + 0.5, b.y);
    }
    for (let y = step; y < doc.sheet.height; y += step) {
      const s = toScreen({ x: 0, y });
      ctx.moveTo(a.x, Math.round(s.y) + 0.5);
      ctx.lineTo(b.x, Math.round(s.y) + 0.5);
    }
    ctx.stroke();
  }
  ctx.strokeStyle = cssVar('--sheet-border', '#adb5bd');
  ctx.strokeRect(a.x + 0.5, a.y + 0.5, b.x - a.x, b.y - a.y);

  for (const e of doc.entities) {
    if (!isVisible(e)) continue;
    const color = layerColor(e.layer);
    if (selection.has(e.id)) drawEntity(e, '#f59f00', 3);
    else if (hover === e.id) drawEntity(e, color, 2.6);
    else drawEntity(e, color, 1.6);
  }

  const preview = previewEntity();
  if (preview) drawEntity(preview, layerColor(doc.activeLayer), 1.4, true);

  if (drag?.mode === 'box') {
    const s = toScreen(drag.start);
    const t = toScreen(drag.end);
    ctx.fillStyle = 'rgba(28,126,214,0.08)';
    ctx.strokeStyle = '#1c7ed6';
    ctx.setLineDash([4, 3]);
    ctx.fillRect(s.x, s.y, t.x - s.x, t.y - s.y);
    ctx.strokeRect(s.x, s.y, t.x - s.x, t.y - s.y);
    ctx.setLineDash([]);
  }

  if (snapped?.onEntity && tool !== 'pan') {
    const s = toScreen(snapped.p);
    ctx.strokeStyle = '#e8590c';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(s.x - 5, s.y - 5, 10, 10);
  }

  const p = snapped?.p ?? cursor;
  $('coords').textContent = `x ${p.x.toFixed(2).replace('.', ',')} · y ${(doc.sheet.height - p.y).toFixed(2).replace('.', ',')} mm`;
  $('zoom').textContent = `${Math.round(view.scale * 100) / 100} px/mm`;
}

// ------------------------------------------------------------ edição

function addEntity(partial: Omit<Entity, 'id' | 'layer'>): void {
  commit((d) => {
    d.entities.push({ ...partial, id: newId(d), layer: d.activeLayer });
  });
}

function finishDraft(close = false): void {
  if (tool === 'polyline' && draft.length >= 2) {
    addEntity({ kind: 'polyline', points: draft, closed: close && draft.length >= 3 });
  }
  draft = [];
  render();
}

function deleteSelection(): void {
  if (!selection.size) return;
  commit((d) => {
    for (const e of d.entities) if (selection.has(e.id)) detachFromPaper(d, e);
    d.entities = d.entities.filter((e) => !selection.has(e.id));
  });
  selection.clear();
  afterChange();
}

function nudge(dx: number, dy: number): void {
  if (!selection.size) return;
  commit((d) => {
    d.entities = d.entities.map((e) => {
      if (!selection.has(e.id)) return e;
      detachFromPaper(d, e);
      return translate(e, dx, dy);
    });
  });
}

function setTool(next: Tool): void {
  tool = next;
  draft = [];
  drag = null;
  for (const btn of document.querySelectorAll<HTMLButtonElement>('#tools button')) {
    btn.setAttribute('aria-pressed', String(btn.dataset.tool === next));
  }
  $('hint').textContent = TOOLS.find((t) => t.id === next)!.hint;
  canvas.style.cursor = next === 'pan' ? 'grab' : next === 'select' ? 'default' : 'crosshair';
  render();
}

// ------------------------------------------------------------- mouse

function eventPoint(ev: MouseEvent): { sx: number; sy: number; p: Pt } {
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  return { sx, sy, p: toWorld(sx, sy) };
}

canvas.addEventListener('pointerdown', (ev) => {
  canvas.focus();
  canvas.setPointerCapture(ev.pointerId);
  const { sx, sy, p } = eventPoint(ev);
  if (ev.button === 1 || tool === 'pan' || spaceDown) {
    drag = { mode: 'pan', sx, sy, ox: view.ox, oy: view.oy };
    canvas.style.cursor = 'grabbing';
    return;
  }
  if (ev.button === 2) {
    if (tool === 'polyline') finishDraft();
    else draft = [];
    render();
    return;
  }
  if (ev.button !== 0) return;
  if (tool === 'select') {
    const hit = hitTest(p);
    if (hit) {
      if (ev.shiftKey) {
        if (selection.has(hit.id)) selection.delete(hit.id);
        else selection.add(hit.id);
      } else if (!selection.has(hit.id)) {
        selection = new Set([hit.id]);
      }
      const originals = doc.entities.filter((e) => selection.has(e.id) && isEditable(e));
      drag = { mode: 'move', start: snap(p, ev.altKey).p, originals: structuredClone(originals), moved: false };
    } else {
      if (!ev.shiftKey) selection.clear();
      drag = { mode: 'box', start: p, end: p, additive: ev.shiftKey };
    }
    refreshPanels();
    render();
    return;
  }
  const point = ortho(snap(p, ev.altKey).p, ev.shiftKey);
  if (tool === 'polyline') {
    if (draft.length >= 3 && dist(point, draft[0]) * view.scale <= SNAP_PX) {
      finishDraft(true);
      return;
    }
    if (ev.detail >= 2) {
      finishDraft();
      return;
    }
    const last = draft[draft.length - 1];
    if (!last || dist(last, point) > 1e-6) draft.push(point);
  } else if (!draft.length) {
    draft = [point];
  } else {
    const start = draft[0];
    if (dist(start, point) > 1e-6) {
      if (tool === 'line') addEntity({ kind: 'line', points: [start, point], closed: false });
      if (tool === 'rect') addEntity({ kind: 'polyline', points: rectPoints(start, point), closed: true });
      if (tool === 'circle') addEntity({ kind: 'circle', points: [], center: start, radius: dist(start, point), closed: true });
    }
    draft = tool === 'line' ? [point] : [];
  }
  render();
});

canvas.addEventListener('pointermove', (ev) => {
  const { sx, sy, p } = eventPoint(ev);
  cursor = p;
  if (drag?.mode === 'pan') {
    view.ox = drag.ox + (sx - drag.sx);
    view.oy = drag.oy + (sy - drag.sy);
    render();
    return;
  }
  if (drag?.mode === 'box') {
    drag.end = p;
    render();
    return;
  }
  if (tool === 'select' && !drag) {
    const hit = hitTest(p);
    hover = hit?.id ?? null;
    canvas.style.cursor = hit ? 'move' : 'default';
    snapped = null;
  } else if (tool !== 'pan') {
    const s = snap(p, ev.altKey);
    snapped = { p: ortho(s.p, ev.shiftKey), onEntity: s.onEntity };
  }
  if (drag?.mode === 'move') {
    const s = snap(p, ev.altKey);
    snapped = s;
    const dx = s.p.x - drag.start.x;
    const dy = s.p.y - drag.start.y;
    if (!drag.moved && Math.hypot(dx, dy) * view.scale < 2) return;
    if (!drag.moved) {
      undoStack.push(cloneDoc(doc));
      redoStack = [];
      drag.moved = true;
    }
    const byId = new Map(drag.originals.map((e) => [e.id, e]));
    doc.entities = doc.entities.map((e) => {
      const original = byId.get(e.id);
      return original ? translate(original, dx, dy) : e;
    });
  }
  render();
});

canvas.addEventListener('pointerup', () => {
  if (drag?.mode === 'move' && drag.moved) {
    for (const e of doc.entities) if (selection.has(e.id)) detachFromPaper(doc, e);
    afterChange();
  }
  if (drag?.mode === 'box') {
    const box = normalizeBox(drag.start, drag.end);
    if (box.maxX - box.minX > 0 || box.maxY - box.minY > 0) {
      const inside = doc.entities.filter((e) => isEditable(e) && boxContains(box, entityBox(e))).map((e) => e.id);
      selection = new Set([...(drag.additive ? selection : []), ...inside]);
    }
    refreshPanels();
  }
  drag = null;
  if (tool === 'pan') canvas.style.cursor = 'grab';
  render();
});

canvas.addEventListener('dblclick', () => {
  if (tool === 'polyline') finishDraft();
});
canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());
canvas.addEventListener(
  'wheel',
  (ev) => {
    ev.preventDefault();
    const { sx, sy } = eventPoint(ev);
    zoomAt(sx, sy, Math.exp(-ev.deltaY * 0.0015));
  },
  { passive: false },
);
canvas.addEventListener('pointerleave', () => {
  hover = null;
  render();
});

// ----------------------------------------------------------- teclado

window.addEventListener('keydown', (ev) => {
  const target = ev.target as HTMLElement;
  if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA') return;
  const mod = ev.ctrlKey || ev.metaKey;
  const key = ev.key.toLowerCase();
  if (mod && key === 'z') {
    ev.preventDefault();
    if (ev.shiftKey) redo();
    else undo();
    return;
  }
  if (mod && key === 'y') {
    ev.preventDefault();
    redo();
    return;
  }
  if (mod && key === 'a') {
    ev.preventDefault();
    selection = new Set(doc.entities.filter(isEditable).map((e) => e.id));
    refreshPanels();
    render();
    return;
  }
  if (mod && key === 's') {
    ev.preventDefault();
    saveFile();
    return;
  }
  if (mod) return;
  if (ev.key === ' ') {
    spaceDown = true;
    ev.preventDefault();
    return;
  }
  if (ev.key === 'Escape') {
    if (tool === 'polyline' && draft.length >= 2) finishDraft();
    else draft = [];
    selection.clear();
    refreshPanels();
    render();
    return;
  }
  if (ev.key === 'Enter') {
    finishDraft();
    return;
  }
  if (ev.key === 'Backspace' && draft.length) {
    draft.pop();
    render();
    return;
  }
  if (ev.key === 'Delete' || ev.key === 'Backspace') {
    deleteSelection();
    return;
  }
  const step = ev.shiftKey ? 10 : 1;
  const arrows: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  if (arrows[ev.key] && selection.size) {
    ev.preventDefault();
    nudge(...arrows[ev.key]);
    return;
  }
  if (key === 'f') {
    fit();
    return;
  }
  const t = TOOLS.find((x) => x.key.toLowerCase() === key);
  if (t) setTool(t.id);
});
window.addEventListener('keyup', (ev) => {
  if (ev.key === ' ') spaceDown = false;
});

// ------------------------------------------------------------ painéis

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, unknown> = {}, children: (Node | string)[] = []): HTMLElementTagNameMap[K] {
  const node: HTMLElementTagNameMap[K] = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children);
  return node;
}

function refreshLayers(): void {
  const list = $('layers');
  list.replaceChildren();
  for (const layer of doc.layers) {
    const count = doc.entities.filter((e) => e.layer === layer.id).length;
    const active = el('input', { type: 'radio', name: 'active-layer', checked: layer.id === doc.activeLayer, title: 'Camada ativa' });
    active.addEventListener('change', () => {
      doc.activeLayer = layer.id;
      save();
    });
    const color = el('input', { type: 'color', value: layer.color, title: 'Cor' });
    color.addEventListener('change', () => commit((d) => (layerOf(d, layer.id)!.color = color.value)));
    const name = el('span', { className: 'layer-name', textContent: `${layer.name} (${count})`, title: 'Duplo clique para renomear' });
    name.addEventListener('dblclick', () => {
      const value = prompt('Nome da camada', layer.name)?.trim();
      if (value) commit((d) => (layerOf(d, layer.id)!.name = value));
    });
    const vis = el('input', { type: 'checkbox', checked: layer.visible, title: 'Visível' });
    vis.addEventListener('change', () => commit((d) => (layerOf(d, layer.id)!.visible = vis.checked)));
    const lock = el('input', { type: 'checkbox', checked: layer.locked, title: 'Travada' });
    lock.addEventListener('change', () => commit((d) => (layerOf(d, layer.id)!.locked = lock.checked)));
    const row = el('li', { className: layer.id === doc.activeLayer ? 'active' : '' }, [
      active,
      color,
      name,
      el('label', { title: 'Visível' }, [vis, 'ver']),
      el('label', { title: 'Travada' }, [lock, 'travar']),
    ]);
    if (layer.id !== DRAW_LAYER && layer.id !== LIVE_LAYER) {
      const remove = el('button', { textContent: '×', title: 'Excluir camada e seus itens', className: 'icon' });
      remove.addEventListener('click', () => {
        if (count && !confirm(`Excluir a camada "${layer.name}" e ${count} itens?`)) return;
        commit((d) => {
          d.layers = d.layers.filter((l) => l.id !== layer.id);
          d.entities = d.entities.filter((e) => e.layer !== layer.id);
          if (d.activeLayer === layer.id) d.activeLayer = DRAW_LAYER;
        });
      });
      row.append(remove);
    }
    list.append(row);
  }
}

function numberInput(label: string, value: number, onChange: (v: number) => void): HTMLLabelElement {
  const input = el('input', { type: 'number', step: '0.1', value: value.toFixed(2) });
  input.addEventListener('change', () => {
    const v = Number(input.value);
    if (Number.isFinite(v)) onChange(v);
  });
  return el('label', { className: 'field' }, [label, input]);
}

function editSelected(mutate: (e: Entity) => void): void {
  const [id] = selection;
  commit((d) => {
    const e = d.entities.find((x) => x.id === id);
    if (!e) return;
    detachFromPaper(d, e);
    mutate(e);
  });
}

function refreshProps(): void {
  const box = $('props');
  box.replaceChildren();
  const items = doc.entities.filter((e) => selection.has(e.id));
  if (!items.length) {
    box.append(el('p', { className: 'muted', textContent: 'Nada selecionado' }));
    return;
  }
  const layerSelect = el('select');
  for (const l of doc.layers) layerSelect.append(el('option', { value: l.id, textContent: l.name }));
  layerSelect.value = items.every((e) => e.layer === items[0].layer) ? items[0].layer : '';
  layerSelect.addEventListener('change', () =>
    commit((d) => {
      for (const e of d.entities) {
        if (!selection.has(e.id)) continue;
        detachFromPaper(d, e);
        e.layer = layerSelect.value;
      }
    }),
  );
  const names = { line: 'Linha', polyline: 'Polilinha', circle: 'Círculo' } as const;
  box.append(el('p', { textContent: items.length === 1 ? names[items[0].kind] : `${items.length} itens` }));
  if (items.some((e) => e.paperId)) box.append(el('p', { className: 'muted', textContent: 'Veio do papel ao vivo. Editar desliga este item do papel.' }));
  box.append(el('label', { className: 'field' }, ['Camada', layerSelect]));
  if (items.length !== 1) return;
  const e = items[0];
  const H = doc.sheet.height;
  if (e.kind === 'circle' && e.center && e.radius !== undefined) {
    const c = e.center;
    box.append(
      numberInput('Centro x', c.x, (v) => editSelected((x) => (x.center = { ...x.center!, x: v }))),
      numberInput('Centro y', H - c.y, (v) => editSelected((x) => (x.center = { ...x.center!, y: H - v }))),
      numberInput('Raio', e.radius, (v) => v > 0 && editSelected((x) => (x.radius = v))),
    );
  } else if (e.kind === 'line') {
    e.points.forEach((p, i) => {
      box.append(
        numberInput(`x${i + 1}`, p.x, (v) => editSelected((x) => (x.points[i] = { ...x.points[i], x: v }))),
        numberInput(`y${i + 1}`, H - p.y, (v) => editSelected((x) => (x.points[i] = { ...x.points[i], y: H - v }))),
      );
    });
  } else {
    const closed = el('input', { type: 'checkbox', checked: e.closed });
    closed.addEventListener('change', () => editSelected((x) => (x.closed = closed.checked && x.points.length >= 3)));
    box.append(el('p', { textContent: `${e.points.length} vértices` }), el('label', { className: 'field' }, ['Fechada', closed]));
  }
  box.append(el('p', { className: 'muted', textContent: `Comprimento ${entityLength(e).toFixed(2).replace('.', ',')} mm` }));
}

function refreshPanels(): void {
  refreshLayers();
  refreshProps();
}

// ------------------------------------------------------- arquivos

function download(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function saveFile(): void {
  download('desenho.draw-cad.json', JSON.stringify(doc, null, 2), 'application/json');
}

$('new').addEventListener('click', () => {
  if (doc.entities.length && !confirm('Começar um desenho novo? O atual sai da tela (salve antes se quiser).')) return;
  commit((d) => {
    const fresh = newDoc();
    Object.assign(d, fresh, { paper: { ...fresh.paper, session: d.paper.session, page: d.paper.page } });
  });
  if (lastPaper) {
    doc.paper.importedPages = lastPaper.history.map((h) => h.page);
    applyPaperState(doc, lastPaper);
    afterChange();
  }
});
$('open').addEventListener('click', () => $<HTMLInputElement>('file').click());
$<HTMLInputElement>('file').addEventListener('change', async (ev) => {
  const input = ev.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  try {
    const opened = parseDoc(await file.text());
    commit((d) => Object.assign(d, opened));
    fit();
  } catch (error) {
    alert(`Não consegui abrir: ${(error as Error).message}`);
  }
});
$('save').addEventListener('click', saveFile);
$('export-dxf').addEventListener('click', () => download('desenho.dxf', toDxf(doc), 'application/dxf'));
$('export-svg').addEventListener('click', () => download('desenho.svg', toSvg(doc), 'image/svg+xml'));
$('undo').addEventListener('click', undo);
$('redo').addEventListener('click', redo);
$('delete').addEventListener('click', deleteSelection);
$('fit').addEventListener('click', fit);
$('theme').addEventListener('click', () => {
  theme = nextTheme(theme);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* sem armazenamento: vale só nesta aba */
  }
  applyTheme();
});
darkQuery.addEventListener('change', () => theme === 'system' && render());
$('add-layer').addEventListener('click', () => {
  const name = prompt('Nome da nova camada', `Camada ${doc.layers.length + 1}`)?.trim();
  if (!name) return;
  commit((d) => {
    d.activeLayer = addLayer(d, name).id;
  });
});

for (const t of TOOLS) {
  const btn = el('button', { textContent: t.label, title: `${t.label} (${t.key})` });
  btn.dataset.tool = t.id;
  btn.addEventListener('click', () => setTool(t.id));
  $('tools').append(btn);
}

// ---------------------------------------------------- papel ao vivo

function connectPaper(): void {
  const live = $('live');
  const source = new EventSource('api/events');
  source.addEventListener('state', (ev) => {
    lastPaper = JSON.parse((ev as MessageEvent).data) as PaperState;
    const shouldFit = doc.paper.session === null;
    if (applyPaperState(doc, lastPaper)) afterChange();
    if (shouldFit) fit();
    live.className = 'live on';
    live.textContent = `Ao vivo · página ${lastPaper.page} · ${lastPaper.strokes.length} traços`;
  });
  source.addEventListener('error', () => {
    live.className = 'live off';
    live.textContent = 'Desconectado — rode python main.py --cad';
  });
}

new ResizeObserver(resize).observe(canvas);
applyTheme();
setTool('select');
refreshPanels();
resize();
fit();
connectPaper();
