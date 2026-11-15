const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('physical-ai-studio/application/backend/src/so101_architecture/ui_extension.js', 'utf8');
const start = source.indexOf('  let inspectionRequestActive');
const end = source.indexOf('  function stopCamRefresh()', start);
function harness(payload, fails = false) {
  const elements = {};
  for (const camera of ['plate', 'wrist']) {
    elements[`so101-cam-img-${camera}`] = {src: 'old-overlay'};
    elements[`so101-inspection-${camera}`] = {textContent: '', style: {}};
  }
  const context = vm.createContext({document: {getElementById: id => elements[id]},
    setTimeout, clearTimeout, AbortController, Date,
    fetch: async () => {if (fails) throw Error('offline'); return {ok: true, json: async () => payload};},
  });
  vm.runInContext(source.slice(start, end), context);
  return {elements, run: () => vm.runInContext('refreshInspectionEvidence()', context)};
}
test('fresh inspection packet replaces camera with its paired overlay', async () => {
  const h = harness({cameras: {plate: {fresh:true, image_jpeg:'source', overlay_jpeg:'paired', decision:'UNCALIBRATED', reason:'preview', frame_id:'123'}}});
  await h.run();
  assert.equal(h.elements['so101-cam-img-plate'].src, 'data:image/jpeg;base64,paired');
  assert.match(h.elements['so101-inspection-plate'].textContent, /UNCALIBRATED.*123/);
  assert.match(h.elements['so101-inspection-wrist'].textContent, /not connected/);
});
test('stale and failed evidence remove old heatmaps instead of implying detection', async () => {
  for (const failed of [false,true]) {
    const h = harness({cameras:{plate:{fresh:false, decision:'STALE', reason:'expired'}}}, failed);
    await h.run();
    assert.match(h.elements['so101-cam-img-plate'].src, /^\/api\/so101\/camera\/plate.jpg/);
    assert.match(h.elements['so101-inspection-plate'].textContent, failed ? /unavailable/ : /STALE/);
  }
});
