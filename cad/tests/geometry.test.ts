import { test } from 'node:test';
import assert from 'node:assert/strict';
import { distToEntity, entityBox, entityLength, rectPoints, snapPoints, snapToGrid, translate, type Entity } from '../src/geometry.ts';

const line: Entity = { id: 'a', layer: 'l', kind: 'line', points: [{ x: 0, y: 0 }, { x: 10, y: 0 }], closed: false };
const circle: Entity = { id: 'c', layer: 'l', kind: 'circle', points: [], closed: true, center: { x: 5, y: 5 }, radius: 2 };

test('distância a linha e círculo', () => {
  assert.equal(distToEntity({ x: 5, y: 3 }, line), 3);
  assert.equal(distToEntity({ x: -4, y: 3 }, line), 5);
  assert.equal(distToEntity({ x: 5, y: 9 }, circle), 2);
});

test('caixa, comprimento e translação', () => {
  assert.deepEqual(entityBox(circle), { minX: 3, minY: 3, maxX: 7, maxY: 7 });
  const rect: Entity = { id: 'r', layer: 'l', kind: 'polyline', points: rectPoints({ x: 0, y: 0 }, { x: 4, y: 3 }), closed: true };
  assert.equal(entityLength(rect), 14);
  const moved = translate(circle, 1, -1);
  assert.deepEqual(moved.center, { x: 6, y: 4 });
  assert.deepEqual(circle.center, { x: 5, y: 5 });
});

test('snap em pontos notáveis e grade', () => {
  assert.deepEqual(snapPoints(line).at(-1), { x: 5, y: 0 });
  assert.deepEqual(snapPoints(circle)[0], { x: 5, y: 5 });
  assert.deepEqual(snapToGrid({ x: 2.4, y: 7.6 }, 5), { x: 0, y: 10 });
});
