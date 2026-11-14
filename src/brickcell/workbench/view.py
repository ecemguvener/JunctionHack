"""Self-contained offline HTML. Untrusted metadata is text, never executable markup."""

import html
import json
from pathlib import Path

CSS = """
:root{color-scheme:dark;font:16px system-ui;background:#101821;color:#e7eef5}
body{max-width:1180px;margin:32px auto;padding:0 24px}h1{font-size:32px;margin-bottom:8px}
h2{font-size:22px}p{line-height:1.6;color:#b9c9d8}a{color:#71d5c4}button,select,input{
font:inherit;border:1px solid #506274;border-radius:6px;background:#203140;color:inherit;padding:8px}
button{cursor:pointer}input[type=range]{width:95%;padding:0}section,.card{background:#182532;
border:1px solid #314655;border-radius:10px;padding:20px;margin:18px 0}.grid{
display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}
img{max-width:100%;background:#080d13;object-fit:contain;border-radius:6px}
.camera{width:100%;height:280px;image-rendering:auto}.badge{display:inline-block;padding:6px 10px;
background:#244449;border-radius:6px;margin:4px}.warn{color:#ffd291}.bad{color:#ffacac}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #344453}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}li{margin:8px 0;overflow-wrap:anywhere}
small{color:#bdccda}canvas{width:100%;height:180px;background:#0c151e}summary{cursor:pointer}
@media(max-width:600px){body{padding:0 12px}h1{font-size:26px}.camera{height:220px}td,th{padding:5px;font-size:12px}}
"""


def page(path: Path, title: str, body: str, data: dict | None = None, script: str = "") -> None:
    payload = (
        json.dumps(data or {}, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    )
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><small>BRICKCELL · OFFLINE DATA WORKBENCH</small>
<h1>{html.escape(title)}</h1>{body}
<script type="application/json" id="data">{payload}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
{script}</script></body></html>""")


def summary(report: dict) -> str:
    status = (
        "READY FOR A SOFTWARE SMOKE RUN" if report["smoke_ready"] else "NOT READY — REVIEW FINDINGS"
    )

    def items(values):
        return "".join(f"<li>{html.escape(str(x))}</li>" for x in values)

    return f"""<section><span class="badge">{status}</span>
<p>Dataset checks only. Physical deployment and reliable learned movement are not validated.</p>
<details open><summary>Blocking findings ({len(report["errors"])})</summary>
<ul class="bad">{items(report["errors"]) or "<li>None in this validation scope.</li>"}</ul></details>
<details open><summary>Scope and warnings</summary><ul class="warn">{items(report["warnings"])}</ul></details>
<p>Partitions: {html.escape(json.dumps(report["counts"]))}</p></section>"""


def policy_page(path: Path, report: dict, episodes: list) -> None:
    source = (
        "SYNTHETIC DATA · SIMULATION / FIXTURE"
        if report["synthetic"]
        else "REAL DATA · RECORDED, NOT LIVE"
    )
    page(
        path,
        "Demonstration review",
        f"""<p>{source} · Movement source: {html.escape(str(report["movement_source"]))}</p>
<p>Review camera frames and observed joints against the commanded targets at the same recorded index.
This page cannot operate a robot. <a href="report.json">Download validation JSON</a></p>
{summary(report)}
<section><h2>Training versus rollout evidence</h2>
<p>Supplied reports are historical evidence, not rerun measurements. Dataset readiness does not imply policy success.
No automatic association with a checkpoint is claimed.</p>
<details><summary>Training report</summary><pre>{html.escape(json.dumps(report.get("training_evidence"), indent=2))}</pre></details>
<details open><summary>Closed-loop rollout report (null means not supplied)</summary>
<pre>{html.escape(json.dumps(report.get("rollout_evidence"), indent=2))}</pre></details></section>
<section><label>Episode <select id="episode" aria-label="Episode"></select></label>
<p id="episodeInfo"></p><ul class="bad" id="issues"></ul>
<div class="grid"><div><h2>Overview camera</h2><img class="camera" id="overview" alt="Overview frame"><p id="overviewStatus"></p></div>
<div><h2>Wrist camera</h2><img class="camera" id="wrist" alt="Wrist frame"><p id="wristStatus"></p></div></div>
<button id="play">Play</button> <button id="previous">Previous</button> <button id="next">Next</button>
<p id="time"></p><input id="frame" aria-label="Frame" type="range" min="0" value="0" step="1">
<table><thead><tr><th>Joint</th><th>Observed</th><th>Commanded</th><th>Difference</th></tr></thead><tbody id="joints"></tbody></table>
<p>Units: {html.escape(str(report["units"]))}. Numerical limit tolerance: 1e-6 radians. Red cells exceed the supplied profile. Values are never clipped.</p>
<label>Plot joint <select id="joint" aria-label="Plot joint"></select></label>
<p>Observed: mint · commanded: orange · white cursor: selected frame</p><canvas id="plot" width="1000" height="180"></canvas>
</section><section><h2>Next commands</h2><pre>{html.escape(chr(10).join(report.get("commands", [])))}</pre>
<p>Commands are suggestions only; this page never starts training.</p></section>""",
        {"report": report, "episodes": episodes},
        POLICY_JS,
    )


POLICY_JS = """
const $=id=>document.getElementById(id), selector=$('episode'), slider=$('frame');
let playing=false,last=0;
data.episodes.forEach((e,i)=>selector.add(new Option(e.episode_id+' · '+e.split,i)));
data.report.joint_names.forEach((n,i)=>$('joint').add(new Option(n,i)));
function episode(){return data.episodes[Number(selector.value)]}
function fmt(x){return Number.isFinite(x)?x.toFixed(5):'MISSING / INVALID'}
function step(delta){slider.value=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+delta));render()}
function render(){
 const e=episode(); if(!e){$('episodeInfo').textContent='No readable episodes';return}
 const p=e.preview,i=Number(slider.value);$('episodeInfo').textContent=e.frames+' recorded frames · '+e.split;
 $('issues').replaceChildren(...e.issues.map(t=>{const li=document.createElement('li');li.textContent=t;return li}));
 $('play').disabled=!p;$('previous').disabled=!p;$('next').disabled=!p;slider.disabled=!p;
 for(const c of ['overview','wrist']){const uri=p?.cameras[c]?.[i];$(c).hidden=!uri;if(uri)$(c).src=uri;
 $(c+'Status').textContent=uri?'Recorded frame '+i:'MISSING / INVALID FRAME';}
 $('time').textContent=p?'Frame '+i+' / '+(e.frames-1)+' · timestamp '+fmt(p.timestamp[i])+' s':'Episode cannot be previewed; read findings above.';
 $('joints').replaceChildren();
 if(p) data.report.joint_names.forEach((n,j)=>{const tr=document.createElement('tr');
 const s=p.state[i][j],a=p.action[i][j];[n,fmt(s),fmt(a),Number.isFinite(s)&&Number.isFinite(a)?fmt(a-s):'INVALID'].forEach((v,k)=>{
 const td=document.createElement('td');td.textContent=v;
 if(k===1||k===2){const kind=k===1?'state':'action',bounds=data.report.limits?.joints[n][kind],val=k===1?s:a;
 if(!Number.isFinite(val)||(bounds&&(val<bounds[0]-1e-6||val>bounds[1]+1e-6)))td.className='bad';}tr.append(td)});$('joints').append(tr)});
 const canvas=$('plot'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,1000,180);if(!p)return;
 const j=Number($('joint').value),values=[...p.state,...p.action].map(r=>r[j]).filter(Number.isFinite);
 const lo=Math.min(...values)-0.05,hi=Math.max(...values)+0.05;
 for(const [key,color] of [['state','#71d5c4'],['action','#ffca87']]){ctx.beginPath();ctx.strokeStyle=color;let started=false;
 p[key].forEach((r,k)=>{if(!Number.isFinite(r[j])){started=false;return}const x=k*1000/Math.max(1,e.frames-1),y=170-(r[j]-lo)*160/(hi-lo);
 if(started)ctx.lineTo(x,y);else ctx.moveTo(x,y);started=true});ctx.stroke()}
 ctx.strokeStyle='white';ctx.beginPath();const x=i*1000/Math.max(1,e.frames-1);ctx.moveTo(x,0);ctx.lineTo(x,180);ctx.stroke();
}
function choose(){playing=false;$('play').textContent='Play';slider.value=0;slider.max=Math.max(0,(episode()?.frames||1)-1);render()}
selector.onchange=choose;slider.oninput=render;$('joint').onchange=render;
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';last=performance.now()};
function tick(now){if(playing&&now-last>=1000/data.report.fps){step(1);last=now;
 if(Number(slider.value)>=Number(slider.max)){playing=false;$('play').textContent='Play'}}requestAnimationFrame(tick)}
choose();requestAnimationFrame(tick);
"""


def inspection_page(path: Path, report: dict, cards: list) -> None:
    page(
        path,
        "Inspection image review",
        f"""<p>Appearance dataset · {report["synthetic_images"]} synthetic / {report["total_images"]} total images.
Recorded files, not a live camera. <a href="report.json">Download validation JSON</a></p>{summary(report)}
<section><h2>Five-face capture coverage</h2><p>Required: front, right, back, left, studded top. Bottom excluded.
Labels complete: <strong>{report["five_face_labels_complete"]}</strong>. This does not verify visible surface area.</p>
<details><summary>Missing metadata</summary><pre>{html.escape(json.dumps(report["missing_metadata"], indent=2))}</pre></details>
<details><summary>Show object/session coverage</summary><div id="coverage"></div></details></section>
<section><label>Filter label <select id="label"><option value="">All labels</option><option>good</option><option>defective</option></select></label>
<label>Filter face <select id="face"><option value="">All faces</option>{"".join("<option>" + f + "</option>" for f in ("front", "right", "back", "left", "top"))}<option value="unknown">Unknown / out of scope</option></select></label>
<label>Filter split <select id="split"><option value="">All splits</option><option>train</option><option>val</option><option>test</option><option>unsplit</option></select></label>
<p id="count"></p><div class="grid" id="images"></div></section>
<section><h2>Next commands</h2><pre>{html.escape(chr(10).join(report.get("commands", [])))}</pre></section>""",
        {"report": report, "cards": cards},
        INSPECTION_JS,
    )


INSPECTION_JS = """
const $=id=>document.getElementById(id);
for(const row of data.report.coverage){const p=document.createElement('p');
 p.textContent=row.object_id+' / '+row.session_id+' — recorded: '+(row.recorded_faces.join(', ')||'none')+'; missing: '+(row.missing_faces.join(', ')||'none');$('coverage').append(p)}
function render(){const selected=data.cards.filter(c=>(!$('label').value||c.label===$('label').value)&&
 (!$('split').value||c.split===$('split').value)&&(!$('face').value||($('face').value==='unknown'?!data.report.required_faces.includes(c.view_id):c.view_id===$('face').value)));
 $('count').textContent=selected.length+' matching images';$('images').replaceChildren();
 for(const c of selected){const card=document.createElement('article');card.className='card';
 if(c.preview){const im=document.createElement('img');im.src=c.preview;im.alt=c.image_id;im.loading='lazy';card.append(im)}
 const h=document.createElement('h2');h.textContent=c.label+' · '+(c.view_id||'UNKNOWN FACE');card.append(h);
 const brief=document.createElement('p');brief.textContent=(c.object_id||'Unknown object')+' · '+(c.family||'Unknown family')+' · '+(c.color||'Unknown color')+' · '+c.split;card.append(brief);const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent='Capture metadata';details.append(heading);const p=document.createElement('pre');p.textContent=JSON.stringify({...c,preview:undefined},null,2);details.append(p);card.append(details);$('images').append(card)}}
for(const id of ['label','face','split'])$(id).onchange=render;render();
"""
