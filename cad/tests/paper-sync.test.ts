import { test } from 'node:test';
import assert from 'node:assert/strict';
import { newDoc, LIVE_LAYER, type PaperState } from '../src/model.ts';
import { applyPaperState, detachFromPaper } from '../src/paper-sync.ts';

const state = (over: Partial<PaperState> = {}): PaperState => ({
  session: 'S1',
  version: 1,
  page: 1,
  strokes: [
    { id: 's0001', kind: 'line', points: [[10, 10], [50, 10]], closed: false },
    { id: 's0002', kind: 'circle', points: [], closed: true, center: [100, 100], radius: 20 },
  ],
  history: [],
  paper: { format: 'a4', width_mm: 210, height_mm: 297 },
  ...over,
});

test('traços do papel entram na camada ao vivo e atualizam sem duplicar', () => {
  const doc = newDoc();
  assert.equal(applyPaperState(doc, state()), true);
  assert.equal(doc.entities.length, 2);
  assert.ok(doc.entities.every((e) => e.layer === LIVE_LAYER));
  const circleId = doc.entities.find((e) => e.kind === 'circle')!.id;
  const grown = state({ strokes: [{ id: 's0001', kind: 'line', points: [[10, 10], [90, 10]], closed: false }, state().strokes[1]] });
  assert.equal(applyPaperState(doc, grown), true);
  assert.equal(doc.entities.length, 2);
  assert.equal(doc.entities.find((e) => e.kind === 'line')!.points[1].x, 90);
  assert.equal(doc.entities.find((e) => e.kind === 'circle')!.id, circleId);
  assert.equal(applyPaperState(doc, grown), false);
});

test('traço apagado ou editado no CAD não volta', () => {
  const doc = newDoc();
  applyPaperState(doc, state());
  const line = doc.entities.find((e) => e.kind === 'line')!;
  detachFromPaper(doc, line);
  doc.entities = doc.entities.filter((e) => e !== line);
  applyPaperState(doc, state());
  assert.equal(doc.entities.length, 1);
  assert.equal(doc.entities[0].kind, 'circle');
});

test('virada de página congela a página anterior numa camada própria', () => {
  const doc = newDoc();
  applyPaperState(doc, state());
  const page2 = state({ page: 2, strokes: [{ id: 's0001', kind: 'line', points: [[0, 0], [5, 5]], closed: false }], history: [{ page: 1, strokes: state().strokes }] });
  applyPaperState(doc, page2);
  const frozen = doc.layers.find((l) => l.name === 'Papel — página 1')!;
  assert.ok(frozen);
  assert.equal(doc.entities.filter((e) => e.layer === frozen.id).length, 2);
  assert.equal(doc.entities.filter((e) => e.layer === LIVE_LAYER).length, 1);
  assert.ok(doc.entities.filter((e) => e.layer === frozen.id).every((e) => !e.paperId));
  applyPaperState(doc, page2);
  assert.equal(doc.layers.filter((l) => l.name === 'Papel — página 1').length, 1);
});

test('cliente que chega atrasado importa o histórico e a sessão nova congela o que estava ao vivo', () => {
  const doc = newDoc();
  applyPaperState(doc, state({ page: 3, strokes: [], history: [{ page: 1, strokes: state().strokes }, { page: 2, strokes: state().strokes.slice(0, 1) }] }));
  assert.equal(doc.layers.filter((l) => l.name.startsWith('Papel — página')).length, 2);
  assert.equal(doc.entities.length, 3);
  const other = newDoc();
  applyPaperState(other, state());
  applyPaperState(other, state({ session: 'S2', strokes: [] }));
  assert.ok(other.layers.some((l) => l.name === 'Papel (sessão anterior)'));
  assert.equal(other.entities.filter((e) => e.layer === LIVE_LAYER).length, 0);
  assert.equal(other.entities.length, 2);
});

test('item do papel editado no CAD sobrevive à virada de página', () => {
  const doc = newDoc();
  applyPaperState(doc, state());
  const line = doc.entities.find((e) => e.kind === 'line')!;
  detachFromPaper(doc, line);
  line.points[1] = { x: 70, y: 10 };
  applyPaperState(doc, state({ page: 2, strokes: [], history: [] }));
  assert.ok(doc.entities.some((e) => e.id === line.id && e.points[1].x === 70));
});
