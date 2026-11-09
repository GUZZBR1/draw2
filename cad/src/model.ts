import type { Entity } from './geometry.ts';

export type Layer = { id: string; name: string; color: string; visible: boolean; locked: boolean };

export type PaperStroke = {
  id: string;
  kind: 'line' | 'polyline' | 'circle';
  points: [number, number][];
  closed: boolean;
  center?: [number, number];
  radius?: number;
};

export type PaperPage = { page: number; strokes: PaperStroke[] };

export type PaperState = {
  session: string;
  version: number;
  page: number;
  strokes: PaperStroke[];
  history: PaperPage[];
  paper: { format: string; width_mm: number; height_mm: number } | null;
};

export type Doc = {
  format: 'draw-cad';
  version: 1;
  sheet: { width: number; height: number; format: string };
  layers: Layer[];
  entities: Entity[];
  activeLayer: string;
  nextId: number;
  /** Sincronização com o papel: sessão/página espelhadas e ids suprimidos. */
  paper: { session: string | null; page: number; importedPages: number[]; suppressed: string[] };
};

export const LIVE_LAYER = 'papel';
export const DRAW_LAYER = 'desenho';

const PALETTE = ['#e8590c', '#2f9e44', '#9c36b5', '#c2255c', '#0c8599', '#5c940d', '#862e9c'];

export function newDoc(): Doc {
  return {
    format: 'draw-cad',
    version: 1,
    sheet: { width: 210, height: 297, format: 'a4' },
    layers: [
      { id: DRAW_LAYER, name: 'Desenho', color: '#1f2933', visible: true, locked: false },
      { id: LIVE_LAYER, name: 'Papel (ao vivo)', color: '#1c7ed6', visible: true, locked: false },
    ],
    entities: [],
    activeLayer: DRAW_LAYER,
    nextId: 1,
    paper: { session: null, page: 1, importedPages: [], suppressed: [] },
  };
}

export function cloneDoc(doc: Doc): Doc {
  return structuredClone(doc);
}

export function newId(doc: Doc): string {
  return `e${doc.nextId++}`;
}

export function addLayer(doc: Doc, name: string, id?: string): Layer {
  const used = new Set(doc.layers.map((l) => l.id));
  let layerId = id ?? `camada-${doc.layers.length + 1}`;
  let n = 2;
  while (used.has(layerId)) layerId = `${id ?? 'camada'}-${n++}`;
  const layer: Layer = {
    id: layerId,
    name,
    color: PALETTE[doc.layers.length % PALETTE.length],
    visible: true,
    locked: false,
  };
  doc.layers.push(layer);
  return layer;
}

export function layerOf(doc: Doc, id: string): Layer | undefined {
  return doc.layers.find((l) => l.id === id);
}

export function parseDoc(text: string): Doc {
  const value = JSON.parse(text) as Partial<Doc>;
  if (value.format !== 'draw-cad' || value.version !== 1 || !Array.isArray(value.entities) || !Array.isArray(value.layers)) {
    throw new Error('arquivo não é um desenho do Draw CAD (format draw-cad, version 1)');
  }
  const base = newDoc();
  return { ...base, ...value, paper: { ...base.paper, ...(value.paper ?? {}) } } as Doc;
}
