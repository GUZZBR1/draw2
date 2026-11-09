import { test } from 'node:test';
import assert from 'node:assert/strict';
import { newDoc, parseDoc } from '../src/model.ts';
import { hexToAci, toDxf } from '../src/dxf.ts';
import { toSvg } from '../src/svg.ts';

function sample() {
  const doc = newDoc();
  doc.entities.push(
    { id: 'e1', layer: 'desenho', kind: 'line', points: [{ x: 10, y: 20 }, { x: 110, y: 20 }], closed: false },
    { id: 'e2', layer: 'desenho', kind: 'circle', points: [], closed: true, center: { x: 50, y: 50 }, radius: 10 },
    { id: 'e3', layer: 'papel', kind: 'polyline', points: [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }], closed: true },
  );
  return doc;
}

test('DXF R12 em mm com Y para cima', () => {
  const dxf = toDxf(sample()).split('\n');
  assert.equal(dxf[dxf.indexOf('$ACADVER') + 2], 'AC1009');
  const line = dxf.indexOf('LINE');
  assert.equal(dxf[line + 6], '277.000');
  const circle = dxf.indexOf('CIRCLE');
  assert.equal(dxf[circle + 6], '247.000');
  assert.equal(dxf[circle + 10], '10.000');
  assert.equal(dxf.filter((l) => l === 'VERTEX').length, 3);
  assert.equal(dxf.at(-2), 'EOF');
});

test('camada invisível não vai para DXF nem SVG', () => {
  const doc = sample();
  doc.layers.find((l) => l.id === 'papel')!.visible = false;
  assert.ok(!toDxf(doc).includes('VERTEX'));
  assert.ok(!toSvg(doc).includes('<polygon'));
});

test('SVG em mm com o tamanho da folha', () => {
  const svg = toSvg(sample());
  assert.match(svg, /width="210mm" height="297mm" viewBox="0 0 210 297"/);
  assert.match(svg, /<circle cx="50" cy="50" r="10"\/>/);
  assert.match(svg, /<polyline points="10,20 110,20"\/>/);
});

test('cor -> ACI e ida e volta do documento', () => {
  assert.equal(hexToAci('#ff0000'), 1);
  assert.equal(hexToAci('#1c7ed6'), 150);
  assert.equal(hexToAci('ruim'), 7);
  const doc = sample();
  assert.deepEqual(parseDoc(JSON.stringify(doc)), doc);
  assert.throws(() => parseDoc('{"format":"outro"}'));
});
