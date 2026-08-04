from __future__ import annotations

import collections
import html
import json
from pathlib import Path
from typing import Any

from .jsonl_io import read_jsonl


def build_summary(source_path: str | Path, translated_path: str | Path, run_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    source = [r for _, r, _ in read_jsonl(source_path)]
    translated = [r for _, r, _ in read_jsonl(translated_path)]
    warning_counts = collections.Counter(w for r in translated for w in r.get("validation_warnings", []))
    return {
        "source_records": len(source), "translated_records": len(translated),
        "missing_records": len(source) - len(translated), "run_stats": run_stats or {},
        "prompt_labels": dict(collections.Counter(str(r.get("prompt_label")) for r in translated)),
        "response_labels": dict(collections.Counter(str(r.get("response_label")) for r in translated)),
        "tags": dict(collections.Counter(str(r.get("tag")) for r in translated)),
        "length_buckets": dict(collections.Counter(str(r.get("length_bucket")) for r in translated)),
        "validation_warnings": dict(warning_counts),
        "input_source_chars": sum(r.get("input_source_chars", 0) for r in translated),
        "output_translation_chars": sum(r.get("output_translation_chars", 0) for r in translated),
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def build_review_html(translated_path: str | Path, output_path: str | Path) -> None:
    rows = [r for _, r, _ in read_jsonl(translated_path)]
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nemotron pilot translation review</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#f5f6f8;color:#17191c}header{position:sticky;top:0;background:#f5f6f8;padding:8px 0;z-index:2}.controls{display:flex;gap:8px;flex-wrap:wrap}input,select,button{padding:8px}article{background:white;border:1px solid #d9dde3;border-radius:8px;padding:14px;margin:12px 0}.meta{font-size:12px;color:#555;display:flex;gap:10px;flex-wrap:wrap}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}.text{white-space:pre-wrap;overflow-wrap:anywhere}.warn{color:#a33}@media(max-width:800px){.cols{grid-template-columns:1fr}}</style></head><body>
<header><h1>Nemotron EN→VI — human review</h1><div class="controls"><input id="q" placeholder="Tìm kiếm"><select id="tag"><option value="">Mọi tag</option><option>generic</option><option>jailbreaking</option></select><select id="cat"><option value="">Mọi category</option></select><select id="status"><option value="">Mọi review</option><option>pending</option><option>approved</option><option>minor_fix</option><option>major_fix</option><option>retranslate</option><option>provider_refusal</option></select><button id="export">Export decisions JSON</button></div><p id="count"></p></header><main id="list"></main>
<script>const rows=__DATA__,stored=JSON.parse(localStorage.getItem('nemotron-review-v2')||'{}'),decisions={};for(const r of rows)decisions[r.record_uid]=stored[r.record_uid]||{status:r.manual_qa_status||'pending',note:r.manual_qa_note||''};const cats=[...new Set(rows.flatMap(r=>String(r.violated_categories||'').split(',').map(x=>x.trim()).filter(Boolean)))].sort();document.querySelector('#cat').innerHTML+='<option>'+cats.map(c=>String(c).replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]))).join('</option><option>')+'</option>';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function render(){const q=document.querySelector('#q').value.toLowerCase(),tag=document.querySelector('#tag').value,cat=document.querySelector('#cat').value,status=document.querySelector('#status').value;const view=rows.filter(r=>(!tag||r.tag===tag)&&(!cat||String(r.violated_categories||'').split(',').map(x=>x.trim()).includes(cat))&&(!status||(decisions[r.record_uid]?.status||'pending')===status)&&(!q||JSON.stringify(r).toLowerCase().includes(q)));document.querySelector('#count').textContent=`${view.length}/${rows.length} record`;document.querySelector('#list').innerHTML=view.map(r=>`<article><div class="meta"><b>#${esc(r.manual_qa_index||'—')}</b><code>${esc(r.record_uid)}</code><b>${esc(r.tag)}</b><span>${esc(r.prompt_label)} → ${esc(r.response_label)}</span><span>${esc(r.length_bucket)} · ${r.source_chars} chars</span><span>${esc(r.violated_categories)}</span><span>QA ${esc(r.manual_qa_score_10??'—')}/10</span></div><div class="cols"><section><h3>English prompt</h3><div class="text">${esc(r.prompt_en)}</div></section><section><h3>Vietnamese prompt</h3><div class="text">${esc(r.prompt_vi)}</div></section><section><h3>English response</h3><div class="text">${esc(r.response_en)}</div></section><section><h3>Vietnamese response</h3><div class="text">${esc(r.response_vi)}</div></section></div><p class="warn">${esc((r.validation_warnings||[]).join(' · '))}</p><label>Review <select data-uid="${esc(r.record_uid)}"><option>pending</option><option>approved</option><option>minor_fix</option><option>major_fix</option><option>retranslate</option><option>provider_refusal</option></select></label><label> Ghi chú <input data-note="${esc(r.record_uid)}" value="${esc(decisions[r.record_uid]?.note||'')}"></label></article>`).join('');document.querySelectorAll('[data-uid]').forEach(x=>{x.value=decisions[x.dataset.uid]?.status||'pending';x.onchange=()=>save(x.dataset.uid,x.value,decisions[x.dataset.uid]?.note||'')});document.querySelectorAll('[data-note]').forEach(x=>x.onchange=()=>save(x.dataset.note,decisions[x.dataset.note]?.status||'pending',x.value))}function save(uid,status,note){decisions[uid]={status,note};localStorage.setItem('nemotron-review-v2',JSON.stringify(decisions))}for(const id of ['q','tag','cat','status'])document.querySelector('#'+id).addEventListener('input',render);document.querySelector('#export').onclick=()=>{const b=new Blob([JSON.stringify(decisions,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='nemotron_review_decisions.json';a.click();URL.revokeObjectURL(a.href)};render();</script></body></html>'''
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__DATA__", payload), encoding="utf-8")
