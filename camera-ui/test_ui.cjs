const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(__dirname + '/index.html', 'utf8');
const code = html.split('<script>')[1].split('</script>')[0].replace(/poll\(\);\s*$/, '');
const elements = {};
const document = {body: {}, getElementById(id) {return elements[id] ||= {dataset: {}};}};
const context = {document, module: {exports: {}}, console};
vm.runInNewContext(code + '\nmodule.exports = {showDetection, tiles};', context);
const {showDetection, tiles} = context.module.exports;
for (const name of ['side', 'overhead', 'wrist']) {
  const classes = new Set();
  tiles[name] = {tile: {classList: {toggle(c, on) {on ? classes.add(c) : classes.delete(c);}}}, classes};
}
showDetection({state: 'malformed', camera: 'side', detail: 'live rule'});
assert(tiles.side.classes.has('alert'));
assert(!tiles.overhead.classes.has('alert'));
assert(!tiles.wrist.classes.has('alert'));
for (const state of ['stale', 'none', 'ok']) {
  showDetection({state, camera: 'side'});
  assert(!tiles.side.classes.has('alert'));
}
showDetection({state: 'malformed', camera: 'side'});
showDetection(null);
assert(!tiles.side.classes.has('alert'));
assert(html.includes('.tile.alert:not(.down) .frame::after'));
console.log('Camera red warning and stale/offline reset checks passed');
