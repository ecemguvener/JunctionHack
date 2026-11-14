const test = require('node:test');
const assert = require('node:assert/strict');
const evidence = require('../static/evidence.js');

const current = {connected: true, drawable: true, evidence: {mode: 'real'}};
test('real results never overlay the simulation', () => {
  assert.equal(evidence.drawable(current, 'sim', true), false);
  assert.equal(evidence.label(current, 'sim'), 'Different scene');
});
test('simulated results never overlay the real scene', () => {
  const sim = {...current, evidence: {mode: 'sim'}};
  assert.equal(evidence.drawable(sim, 'real', true), false);
});
test('unverified legacy results remain hidden', () => {
  const legacy = {connected: true};
  assert.equal(evidence.drawable(legacy, 'real', true), false);
  assert.equal(evidence.label(legacy, 'real'), 'Unverified result');
});
test('expired acquisition is hidden despite publisher heartbeat', () => {
  assert.equal(evidence.drawable({...current, drawable: false}, 'real', true), false);
});
test('lost dashboard connection hides previously fresh boxes', () => {
  assert.equal(evidence.drawable(current, 'real', false), false);
});
test('matching scene displays only explicitly drawable object estimates', () => {
  assert.equal(evidence.drawable(current, 'real', true), true);
  assert.equal(evidence.label(current, 'real'), 'Projected object estimate');
});
