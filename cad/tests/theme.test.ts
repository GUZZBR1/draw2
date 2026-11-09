import { test } from 'node:test';
import assert from 'node:assert/strict';
import { displayColor, luminance, nextTheme, parseTheme } from '../src/theme.ts';

test('cor quase preta vira clara no escuro e o inverso no claro', () => {
  assert.equal(displayColor('#1f2933', true), '#e9ecef');
  assert.equal(displayColor('#1f2933', false), '#1f2933');
  assert.equal(displayColor('#ffffff', false), '#1f2933');
  assert.equal(displayColor('#ffffff', true), '#ffffff');
  assert.equal(displayColor('#1c7ed6', true), '#1c7ed6');
  assert.equal(displayColor('ruim', true), 'ruim');
});

test('luminância e ciclo do tema', () => {
  assert.equal(luminance('#000000'), 0);
  assert.equal(Math.round(luminance('#ffffff')! * 1000), 1000);
  assert.equal(nextTheme('system'), 'light');
  assert.equal(nextTheme('light'), 'dark');
  assert.equal(nextTheme('dark'), 'system');
  assert.equal(parseTheme('dark'), 'dark');
  assert.equal(parseTheme(null), 'system');
  assert.equal(parseTheme('xyz'), 'system');
});
