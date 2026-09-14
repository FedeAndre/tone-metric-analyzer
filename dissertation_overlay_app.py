from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import dissertation_app as base
from dissertation_app import APP_VERSION, app
from tone_metric.canonical_score import build_hits_from_canonical_score
from tone_metric.musicxml import extract_visual_groups, parse_musicxml
from tone_metric.omr import pdf_to_annotations, pdf_to_musicxml
from tone_metric.pdfview import render_pdf_pages
from tone_metric.physical import build_normalized_overlay


# This module is deliberately presentation-only. The validated analytical endpoint
# remains base.analyze_endpoint; this wrapper only preserves a copy of the uploaded PDF
# and builds a separate score-overlay manifest after the analytical response succeeds.
CACHE = Path('/tmp/tone-metric-dissertation-visual')
CACHE.mkdir(parents=True, exist_ok=True)
VISUAL_JOBS: dict[str, dict] = {}
VISUAL_LOCK = threading.Lock()


def _set_job(session_id: str, **values) -> None:
    with VISUAL_LOCK:
        row = VISUAL_JOBS.setdefault(session_id, {})
        row.update(values)


def _get_job(session_id: str) -> dict | None:
    with VISUAL_LOCK:
        row = VISUAL_JOBS.get(session_id)
        return dict(row) if row is not None else None


def _cleanup_cache(max_age_hours: float = 24.0) -> None:
    cutoff = time.time() - max_age_hours * 3600.0
    for p in CACHE.iterdir():
        try:
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _build_visual_job(session_id: str, pdf_path: Path, initial_meter: str, analysis_payload: dict) -> None:
    session = CACHE / session_id
    try:
        _set_job(session_id, status='processing', stage='Rendering uploaded PDF pages')
        pages = render_pdf_pages(pdf_path, session / 'pages')
        public_pages = [
            {
                **p,
                'url': f'/api/visual-session/{session_id}/page/{p["index"]}',
            }
            for p in pages
        ]
        _set_job(session_id, pages=public_pages, stage='Recovering score geometry with Audiveris')

        symbolic, omr_path = pdf_to_musicxml(pdf_path, session / 'audiveris')
        override = base._meter_override(initial_meter)
        hits, measures, parser_warnings = parse_musicxml(symbolic, initial_meter_override=override)

        # Build only registration metadata from the saved Audiveris score graph.
        # The recovered hits are intentionally ignored: the analytical result above
        # has already been produced by the unchanged dissertation pipeline.
        _unused_hits, canonical_warnings, canonical_meta = build_hits_from_canonical_score(
            omr_path, measures, symbolic_hits=hits
        )

        _set_job(session_id, stage='Building physical page registration')
        annotation_archive, annotation_warning = pdf_to_annotations(pdf_path, session / 'annotations')
        if annotation_archive is None:
            raise RuntimeError(annotation_warning or 'Audiveris did not produce a physical annotation archive.')

        visual_groups, layout_known = extract_visual_groups(
            symbolic, initial_meter_override=override
        )

        # Feed a deep copy of the already-finished Levels result into the read-only
        # registration layer. Nothing here can alter the analytical response/body.
        visual_levels = deepcopy(analysis_payload.get('levels') or {})
        visual_levels['canonical_score_meta'] = canonical_meta
        overlay = build_normalized_overlay(
            annotation_archive,
            visual_groups,
            layout_known,
            visual_levels,
            session / 'physical',
            omr_path=omr_path,
        )
        warnings = list(overlay.get('warnings', []) or [])
        warnings.extend(str(w) for w in canonical_warnings if str(w).strip())
        if annotation_warning:
            warnings.append(str(annotation_warning))

        manifest = {
            'status': 'ready',
            'session_id': session_id,
            'pages': public_pages,
            'overlay': overlay,
            'warnings': list(dict.fromkeys(warnings)),
            'analysis_unchanged': True,
            'registration_source': 'separate-presentation-layer',
        }
        (session / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        _set_job(session_id, **manifest)
    except Exception as exc:
        _set_job(
            session_id,
            status='failed',
            stage='Visual score overlay stopped',
            detail=str(exc),
            analysis_unchanged=True,
        )


# Replace only the HTTP route object. The analytical implementation function in
# dissertation_app.py is left untouched and is called directly below.
for route in list(app.router.routes):
    if getattr(route, 'path', None) == '/api/analyze' and 'POST' in (getattr(route, 'methods', set()) or set()):
        app.router.routes.remove(route)


@app.post('/api/analyze')
async def analyze_with_separate_visual_layer(
    file: UploadFile = File(...),
    initial_meter: str = Form('auto'),
):
    filename = Path(file.filename or 'score').name
    suffix = Path(filename).suffix.lower()

    # Non-PDF inputs retain the original endpoint behavior exactly and do not create
    # a score-image session because there is no uploaded PDF to display.
    if suffix != '.pdf':
        return await base.analyze_endpoint(file=file, initial_meter=initial_meter)

    _cleanup_cache()
    payload = await file.read()
    if not payload:
        raise HTTPException(400, 'The uploaded file is empty.')
    if len(payload) > 80 * 1024 * 1024:
        raise HTTPException(413, 'Upload is larger than the 80 MB online limit.')

    session_id = uuid.uuid4().hex
    session = CACHE / session_id
    session.mkdir(parents=True, exist_ok=True)
    saved_pdf = session / filename
    saved_pdf.write_bytes(payload)

    # Rewind and invoke the existing validated analytical function unchanged.
    await file.seek(0)
    try:
        response = await base.analyze_endpoint(file=file, initial_meter=initial_meter)
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        raise

    try:
        analysis_payload = json.loads(response.body.decode('utf-8'))
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        return response

    _set_job(
        session_id,
        status='queued',
        stage='Waiting to build full-score visualization',
        created_at=time.time(),
        pages=[],
        analysis_unchanged=True,
    )
    threading.Thread(
        target=_build_visual_job,
        args=(session_id, saved_pdf, initial_meter, analysis_payload),
        daemon=True,
        name=f'tone-metric-score-overlay-{session_id[:8]}',
    ).start()

    # The body is byte-for-byte the analytical response produced by base.analyze_endpoint.
    # Only a response header identifies the independent visualization session.
    response.headers['X-Tone-Metric-Visual-Session'] = session_id
    return response


@app.get('/api/visual-session/{session_id}/manifest')
def visual_manifest(session_id: str):
    if not session_id.isalnum():
        raise HTTPException(404, 'Visual score session not found.')
    job = _get_job(session_id)
    if job is None:
        raise HTTPException(404, 'Visual score session not found.')
    return job


@app.get('/api/visual-session/{session_id}/page/{page_index}')
def visual_page(session_id: str, page_index: int):
    if not session_id.isalnum() or page_index < 0:
        raise HTTPException(404, 'Score page not found.')
    path = CACHE / session_id / 'pages' / f'page_{page_index + 1}.png'
    if not path.exists():
        raise HTTPException(404, 'Score page not found.')
    return FileResponse(path, media_type='image/png')


# Add the full-score viewer to the existing page without replacing any analytical UI.
# The existing JSON Download button continues to serialize only the unchanged analysis body.
_EXTRA_STYLE = r'''<style>
.tm-full-score{margin-top:14px}.tm-score-status{margin:8px 0 12px;color:#555;white-space:pre-wrap}.tm-score-pages{display:grid;gap:18px}.tm-score-page{position:relative;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}.tm-score-page img{display:block;width:100%;height:auto}.tm-score-page svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.tm-score-label{font:700 10px Arial,sans-serif;fill:#111;paint-order:stroke;stroke:#fff;stroke-width:2.5px;stroke-linejoin:round}.tm-score-struct{font:italic 9px Arial,sans-serif;fill:#333;paint-order:stroke;stroke:#fff;stroke-width:2px}.tm-score-wave{fill:none;stroke:#111;stroke-width:1.8;opacity:.72}.tm-score-tree{fill:none;stroke:#555;stroke-width:1.2;opacity:.62}.tm-score-pivot{fill:#111;opacity:.65}.tm-score-dot{fill:#111;opacity:.72}
</style>'''

_EXTRA_SCRIPT = r'''<script>
(function(){
 const oldRender=render;
 render=function(d){oldRender(d);tmStartFullScore(window.__tmVisualSession||'');};
 // The existing handler is patched below to capture the response header before render().
 function ensurePanel(){
   let p=document.getElementById('tm-full-score');
   if(p)return p;
   p=document.createElement('div');p.id='tm-full-score';p.className='panel tm-full-score';
   p.innerHTML='<h2>Full uploaded score with analysis</h2><div id="tm-score-status" class="tm-score-status">Waiting for a PDF analysis.</div><div id="tm-score-pages" class="tm-score-pages"></div>';
   document.querySelector('.wrap').appendChild(p);return p;
 }
 function S(tag,a){let e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(a||{}))e.setAttribute(k,v);return e}
 function text(svg,x,y,value,cls){let e=S('text',{x,y,class:cls});e.textContent=value;svg.appendChild(e)}
 function drawPage(page,overlay){
   const wrap=document.createElement('div');wrap.className='tm-score-page';
   const img=document.createElement('img');img.src=page.url;img.alt='Score page '+(page.index+1);wrap.appendChild(img);
   const svg=S('svg',{viewBox:'0 0 1000 1000',preserveAspectRatio:'none'});wrap.appendChild(svg);
   const p=(overlay.pages||[]).find(x=>Number(x.page_index)===Number(page.index))||{};
   const attacks=p.layer_anchors||[], structural=p.structural_anchors||[], waves=p.wave_anchors||[], pivots=p.pivot_anchors||[], nodes=p.tree_nodes||[], branches=p.tree_branches||[];
   for(const a of attacks){const x=1000*Number(a.cx_norm||0),y=1000*Number(a.cy_norm||0);svg.appendChild(S('circle',{cx:x,cy:y,r:3.2,class:'tm-score-dot'}));text(svg,x+5,Math.max(10,y-7),(a.levels||[]).join(','),'tm-score-label')}
   for(const a of structural){const x=1000*Number(a.cx_norm||0),y=1000*Number(a.cy_norm||0);text(svg,x+4,Math.max(10,y-5),'('+(a.levels||[]).join(',')+')','tm-score-struct')}
   const systems=new Map();for(const w of waves){const k=Number(w.system_index||0);if(!systems.has(k))systems.set(k,[]);systems.get(k).push(w)}
   for(const [k,rows] of systems){rows.sort((a,b)=>Number(a.cx_norm)-Number(b.cx_norm));const b=(p.system_bounds||[]).find(z=>Number(z.system_index)===k);if(!b||rows.length<2)continue;const top=1000*Number(b.top_norm||0);const max=Math.max(1,...rows.map(r=>Number(r.height||1)));const pts=rows.map(r=>{const x=1000*Number(r.cx_norm||0);const y=Math.max(8,top-10-(Number(r.height||1)/max)*34);return x+','+y}).join(' ');svg.appendChild(S('polyline',{points:pts,class:'tm-score-wave'}))}
   for(const q of pivots){const x=1000*Number(q.cx_norm||q.x_norm||0),y=1000*Number(q.cy_norm||q.y_norm||0);if(Number.isFinite(x)&&Number.isFinite(y))svg.appendChild(S('circle',{cx:x,cy:y,r:4.2,class:'tm-score-pivot'}))}
   const byNode=new Map(nodes.map(n=>[String(n.node_index),n]));
   for(const b of branches){const a=byNode.get(String(b.source_node_index)),c=byNode.get(String(b.target_node_index));if(!a||!c)continue;svg.appendChild(S('line',{x1:1000*Number(a.cx_norm||0),y1:1000*Number(a.cy_norm||0),x2:1000*Number(c.cx_norm||0),y2:1000*Number(c.cy_norm||0),class:'tm-score-tree'}))}
   return wrap;
 }
 async function tmStartFullScore(session){
   ensurePanel();const st=document.getElementById('tm-score-status'),box=document.getElementById('tm-score-pages');box.innerHTML='';
   if(!session){st.textContent='No PDF visualization session was created for this analysis.';return}
   for(let i=0;i<180;i++){
     try{
       const r=await fetch('/api/visual-session/'+session+'/manifest',{cache:'no-store'}),m=await r.json();
       if(!r.ok)throw Error(m.detail||'Could not read score visualization status.');
       if(m.status==='failed'){st.textContent='The analytical result is complete, but the separate score overlay stopped: '+(m.detail||'unknown error');return}
       if(m.status!=='ready'){st.textContent=(m.stage||'Building score overlay…')+'\nThe analytical result above is already complete and is not being modified.';await new Promise(x=>setTimeout(x,2000));continue}
       const ov=m.overlay||{};st.textContent='Full score visualization ready. The analytical JSON above is unchanged. '+((ov.warnings||[]).length?('Overlay notes: '+ov.warnings.join(' | ')):'');
       for(const page of (m.pages||[]))box.appendChild(drawPage(page,ov));return;
     }catch(e){st.textContent='Waiting for full-score visualization: '+e.message;await new Promise(x=>setTimeout(x,2000))}
   }
   st.textContent='The analytical result is complete. The score overlay is still processing; run the analysis again only if this status never resolves.';
 }
 window.tmStartFullScore=tmStartFullScore;
 ensurePanel();
})();
</script>'''

# Capture the independent session header while leaving the response body untouched.
base.HTML = base.HTML.replace(
    "last=JSON.parse(t);render(last);",
    "window.__tmVisualSession=r.headers.get('X-Tone-Metric-Visual-Session')||'';last=JSON.parse(t);render(last);",
)
base.HTML = base.HTML.replace('</head>', _EXTRA_STYLE + '</head>')
base.HTML = base.HTML.replace('</body>', _EXTRA_SCRIPT + '</body>')
