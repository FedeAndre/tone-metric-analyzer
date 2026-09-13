from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from tone_metric.dissertation_full import analyze_full
from tone_metric.musicxml import parse_musicxml
from tone_metric.omr import pdf_to_musicxml

APP_VERSION = "1.0.0-dissertation"
app = FastAPI(title="Tone-Metric Dissertation Analyzer", version=APP_VERSION)


def _meter_override(value: str | None):
    text = (value or "").strip().lower()
    if not text or text == "auto":
        return None
    try:
        n, d = text.split("/", 1)
        n, d = int(n), int(d)
    except Exception as exc:
        raise ValueError("Meter override must be Auto or n/d (for example 4/4).") from exc
    if n <= 0 or d <= 0:
        raise ValueError("Meter values must be positive.")
    return n, d


def _warnings(result: dict, parser_warnings: list[str]) -> list[str]:
    out = list(parser_warnings)
    levels = result.get("levels", {})
    out.extend(levels.get("tuplet_extension_policy", {}).get("warnings", []) or [])
    for seg in levels.get("segments", []) or []:
        out.extend(seg.get("warnings", []) or [])
    return list(dict.fromkeys(str(x) for x in out if str(x).strip()))


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "pipeline": "levels-waves-pivots-trees",
        "tuplets": ["duplet", "triplet"],
    }


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...), initial_meter: str = Form("auto")):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".mxl", ".musicxml", ".xml"}:
        raise HTTPException(400, "Upload PDF, MXL, MusicXML, or XML.")
    try:
        override = _meter_override(initial_meter)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(payload) > 80 * 1024 * 1024:
        raise HTTPException(413, "Upload is larger than the 80 MB online limit.")

    with tempfile.TemporaryDirectory(prefix="tone-metric-dissertation-") as tmp:
        tmpdir = Path(tmp)
        incoming = tmpdir / filename
        incoming.write_bytes(payload)
        symbolic = incoming
        if suffix == ".pdf":
            try:
                symbolic, _ = pdf_to_musicxml(incoming, tmpdir / "audiveris")
            except Exception as exc:
                raise HTTPException(422, f"PDF optical-music recognition failed: {exc}") from exc
        try:
            hits, measures, parser_warnings = parse_musicxml(symbolic, initial_meter_override=override)
            if not measures:
                raise ValueError("No measures were recovered from the score.")
            result = analyze_full(hits, measures)
        except Exception as exc:
            raise HTTPException(422, f"Tone-metric analysis failed: {exc}") from exc

    result["source"] = {
        "filename": filename,
        "input_type": suffix.lstrip("."),
        "meter_override": "auto" if override is None else f"{override[0]}/{override[1]}",
    }
    result["warnings"] = _warnings(result, parser_warnings)
    return JSONResponse(result)


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tone-Metric Dissertation Analyzer</title><style>
body{font-family:Arial,sans-serif;background:#f5f5f3;color:#171717;margin:0}.wrap{max-width:1180px;margin:auto;padding:24px}.box,.panel,.card{background:white;border:1px solid #ddd;border-radius:14px}.box{padding:22px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin-top:16px}label{display:grid;gap:5px;font-size:12px;font-weight:700}input,select,button{font:inherit;padding:9px;border-radius:8px;border:1px solid #bbb;background:white}button{background:#111;color:white;border-color:#111;font-weight:700;cursor:pointer}.secondary{background:#666;border-color:#666}.status{margin-top:12px;color:#555;white-space:pre-wrap}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px;margin:14px 0}.card{padding:12px}.num{font-size:22px;font-weight:800}.key{font-size:11px;color:#666}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}}.panel{padding:13px}.panel h2{font-size:16px;margin:0 0 8px}svg{width:100%;height:290px;background:#fbfbfa;border:1px solid #eee;border-radius:8px}.scroll{max-height:290px;overflow:auto;font-size:12px}table{border-collapse:collapse;width:100%}th,td{padding:5px;border-bottom:1px solid #eee;text-align:left}.warn{background:#fff7d6;border:1px solid #e6cb66;border-radius:9px;padding:9px;margin-top:8px;font-size:12px}.small{font-size:12px;color:#666}</style></head><body><div class="wrap">
<div class="box"><h1 style="margin:0">Tone-Metric Dissertation Analyzer</h1><p>Levels → Waves → Pivots → Trees. Explicit duplets and triplets are included as documented local binary/ternary extensions and are never inferred from spacing.</p><div class="controls"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Initial meter<select id="meter"><option>Auto</option><option>2/2</option><option>4/4</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze</button><button id="download" class="secondary" disabled>Download JSON</button></div><div id="status" class="status">Ready.</div></div>
<div id="summary" class="cards"></div><div class="grid"><div class="panel"><h2>Levels</h2><svg id="levels" viewBox="0 0 900 290"></svg><div id="levelsMeta" class="small"></div></div><div class="panel"><h2>Wave</h2><svg id="wave" viewBox="0 0 900 290"></svg></div><div class="panel"><h2>Pivots</h2><div id="pivots" class="scroll"></div></div><div class="panel"><h2>Tree</h2><svg id="tree" viewBox="0 0 900 290"></svg></div></div><div id="warnings"></div></div>
<script>
let last=null,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';function E(t,a={}){let x=document.createElementNS(NS,t);for(let[k,v]of Object.entries(a))x.setAttribute(k,v);return x}function C(s){while(s.firstChild)s.removeChild(s.firstChild)}function sample(a,n=450){if(a.length<=n)return a;let q=(a.length-1)/(n-1);return Array.from({length:n},(_,i)=>a[Math.round(i*q)])}function Y(l,m){return 266-(l-1)*238/Math.max(1,m-1)}
function levels(d){let s=$('levels');C(s);let g=(d.levels.segments||[])[0];if(!g)return;let r=sample(g.structural_points||[]),m=g.max_level||1;r.forEach((p,i)=>{let x=16+i*868/Math.max(1,r.length-1);(p.levels||[]).forEach(l=>s.appendChild(E('circle',{cx:x,cy:Y(l,m),r:3,fill:'#111'})))});$('levelsMeta').textContent=`First meter segment · ${r.length} displayed positions · Levels 1–${m}`}
function wave(d){let s=$('wave');C(s);let r=sample(d.waves||[]);if(!r.length)return;let m=Math.max(...r.map(p=>p.height||1)),p=r.map((z,i)=>`${16+i*868/Math.max(1,r.length-1)},${Y(z.height,m)}`).join(' ');s.appendChild(E('polyline',{points:p,fill:'none',stroke:'#111','stroke-width':2}))}
function pivots(d){let r=d.pivots||[];$('pivots').innerHTML=r.length?'<table><tr><th>#</th><th>Type</th><th>Start</th><th>End</th><th>Depth</th></tr>'+r.slice(0,500).map((p,i)=>`<tr><td>${i+1}</td><td>${p.kind||''}</td><td>${p.start_measure_number||''}:${p.start_offset_in_measure_quarter||''}</td><td>${p.end_measure_number||''}:${p.end_offset_in_measure_quarter||''}</td><td>${p.drop_depth??''}</td></tr>`).join('')+'</table>':'No pivots detected.'}
function tree(d){let s=$('tree');C(s),T=d.trees||{},n=(T.nodes||[]).slice(0,450),idx=new Map(n.map((x,i)=>[x.node_index,i]));if(!n.length)return;let m=Math.max(...n.map(x=>x.lowest_level||1));let xy=x=>{let i=idx.get(x.node_index);return[16+i*868/Math.max(1,n.length-1),Y(x.lowest_level,m)]};(T.branches||[]).forEach(b=>{if(!idx.has(b.source_node_index)||!idx.has(b.target_node_index))return;let a=n[idx.get(b.source_node_index)],c=n[idx.get(b.target_node_index)],[x1,y1]=xy(a),[x2,y2]=xy(c);s.appendChild(E('line',{x1,y1,x2,y2,stroke:'#777','stroke-width':1.2}))});n.forEach(x=>{let[a,b]=xy(x);s.appendChild(E('circle',{cx:a,cy:b,r:3.1,fill:'#111'}))})}
function render(d){let s=d.summary||{},F=[['Measures',s.measures],['Attacks',s.attacks],['Max Level',s.max_level],['Wave points',s.wave_points],['Pivots',s.pivots],['Tree nodes',s.tree_nodes],['Duplets',s.duplet_spans],['Triplets',s.triplet_spans]];$('summary').innerHTML=F.map(([k,v])=>`<div class="card"><div class="num">${v??0}</div><div class="key">${k}</div></div>`).join('');levels(d);wave(d);pivots(d);tree(d);$('warnings').innerHTML=(d.warnings||[]).map(w=>`<div class="warn">${String(w).replaceAll('<','&lt;')}</div>`).join('')}
$('go').onclick=async()=>{let f=$('file').files[0];if(!f){$('status').textContent='Choose a score first.';return}let fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value.toLowerCase());$('go').disabled=true;$('status').textContent=f.name.toLowerCase().endsWith('.pdf')?'Running Audiveris and tone-metric analysis…':'Running tone-metric analysis…';try{let r=await fetch('/api/analyze',{method:'POST',body:fd}),t=await r.text();if(!r.ok)throw Error(t);last=JSON.parse(t);render(last);$('download').disabled=false;$('status').textContent='Analysis complete.'}catch(e){$('status').textContent='Analysis stopped: '+e.message}finally{$('go').disabled=false}};$('download').onclick=()=>{if(!last)return;let b=new Blob([JSON.stringify(last,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='tone-metric-dissertation-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)
