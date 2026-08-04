#!/usr/bin/env python3
"""Build the thread-local interactive Nemotron audit fragment."""

from __future__ import annotations

import argparse
import json
import pathlib


TEMPLATE = r'''<div id="nemotron-audit">
  <div class="viz-grid" id="na-stats"></div>
  <div class="viz-controls">
    <label class="form-label">Split
      <select id="na-split" class="form-select">
        <option value="all">Tất cả</option><option value="train">Train</option><option value="val">Val</option><option value="test">Test</option>
      </select>
    </label>
    <label class="form-label">Tìm nhãn
      <input id="na-search" class="form-control" type="search" placeholder="Ví dụ: profanity, sexual…">
    </label>
  </div>
  <div id="na-chart" role="img" aria-label="Phân bố nhãn an toàn theo split"></div>
  <div class="table-responsive">
    <table class="table table-sm">
      <thead><tr><th>Nhãn nguyên tử</th><th class="text-end">Tổng</th><th class="text-end">Train</th><th class="text-end">Val</th><th class="text-end">Test</th><th></th></tr></thead>
      <tbody id="na-body"></tbody>
    </table>
  </div>
  <section id="na-detail" hidden>
    <h3 id="na-detail-name"></h3>
    <div class="viz-row text-small text-muted" id="na-detail-meta"></div>
    <div id="na-samples"></div>
  </section>
</div>
<style>
  #nemotron-audit { color: var(--foreground); }
  #nemotron-audit .viz-grid { margin-bottom: 16px; }
  #nemotron-audit .viz-controls { margin: 12px 0 16px; }
  #na-chart { margin: 8px 0 18px; }
  #na-chart .bar-row { display:grid; grid-template-columns:minmax(170px, 1.3fr) 3fr 64px; align-items:center; gap:10px; margin:7px 0; }
  #na-chart .track { height:14px; background:color-mix(in srgb, var(--muted) 70%, transparent); border-radius:999px; overflow:hidden; }
  #na-chart .fill { height:100%; background:var(--viz-series-1); border-radius:999px; }
  #na-chart .value { text-align:right; font-variant-numeric:tabular-nums; }
  #na-detail { margin-top:20px; }
  #na-samples { display:grid; gap:14px; margin-top:12px; }
  #na-samples article { border-left:3px solid var(--border); padding-left:12px; }
  #na-samples p { white-space:pre-wrap; overflow-wrap:anywhere; margin:6px 0; }
  #na-samples .sample-head { display:flex; gap:10px; flex-wrap:wrap; }
  @media (max-width:520px) { #na-chart .bar-row { grid-template-columns:1fr 48px; } #na-chart .track { grid-column:1 / -1; } }
</style>
<script>
(() => {
  const root = document.getElementById('nemotron-audit');
  const data = __DATA__;
  const cats = Object.entries(data.atomic_categories).map(([name,v]) => ({name,...v}));
  const fmt = n => new Intl.NumberFormat('vi-VN').format(n || 0);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const stats = root.querySelector('#na-stats');
  const splitRows = data.splits;
  stats.innerHTML = [
    ['Tổng bản ghi', data.summary.rows, 'Train + val + test'],
    ['Nhãn nguyên tử', data.summary.atomic_category_count, `${fmt(data.summary.category_combination_count)} tổ hợp nhãn`],
    ['Exact overlap', Object.values(data.leakage).reduce((a,v)=>a+v.shared_exact_prompt_response,0), 'Giữa các split']
  ].map(x => `<div class="card viz-stat"><div class="text-muted">${x[0]}</div><div class="viz-stat-value">${fmt(x[1])}</div><div class="text-small text-muted">${x[2]}</div></div>`).join('');
  const splitSelect = root.querySelector('#na-split');
  const search = root.querySelector('#na-search');
  const tbody = root.querySelector('#na-body');
  const chart = root.querySelector('#na-chart');
  const detail = root.querySelector('#na-detail');
  let selected = '';
  function visible() {
    const q = search.value.trim().toLowerCase();
    return cats.filter(c => c.name.toLowerCase().includes(q)).sort((a,b)=>(b[splitSelect.value === 'all' ? 'total' : splitSelect.value]||0)-(a[splitSelect.value === 'all' ? 'total' : splitSelect.value]||0));
  }
  function render() {
    const rows = visible(); const key = splitSelect.value === 'all' ? 'total' : splitSelect.value;
    const max = Math.max(...rows.map(x=>x[key]||0),1);
    chart.innerHTML = rows.slice(0,10).map(c => `<div class="bar-row"><span>${esc(c.name)}</span><div class="track"><div class="fill" style="width:${((c[key]||0)/max*100).toFixed(2)}%"></div></div><span class="value">${fmt(c[key])}</span></div>`).join('');
    tbody.innerHTML = rows.map(c => `<tr><td>${esc(c.name)}</td><td class="text-end">${fmt(c.total)}</td><td class="text-end">${fmt(c.train)}</td><td class="text-end">${fmt(c.val)}</td><td class="text-end">${fmt(c.test)}</td><td><button type="button" class="btn btn-ghost" data-cat="${esc(c.name)}" aria-pressed="${selected===c.name}">Xem mẫu</button></td></tr>`).join('');
    tbody.querySelectorAll('button').forEach(b => b.addEventListener('click', () => show(b.dataset.cat)));
  }
  function show(name) {
    selected = name; const c = cats.find(x=>x.name===name); detail.hidden = false;
    root.querySelector('#na-detail-name').textContent = name;
    root.querySelector('#na-detail-meta').textContent = `Tổng ${fmt(c.total)} · train ${fmt(c.train)} · val ${fmt(c.val)} · test ${fmt(c.test)} · mẫu được cắt tối đa 700 ký tự mỗi trường`;
    root.querySelector('#na-samples').innerHTML = c.samples.map((s,i) => `<article><div class="sample-head text-small"><span class="viz-badge">${esc(s.split)}</span><code>${esc(s.id)}</code><span>${esc(s.prompt_label)} → ${esc(s.response_label)}</span></div><div class="text-small text-muted">${esc(s.all_categories)}</div><p><strong>Prompt:</strong> ${esc(s.prompt)||'<em>trống</em>'}</p><p><strong>Response:</strong> ${esc(s.response)||'<em>trống</em>'}</p></article>`).join('');
    render(); detail.scrollIntoView({behavior:'smooth',block:'start'});
  }
  splitSelect.addEventListener('change', render); search.addEventListener('input', render); render();
})();
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    data = json.loads(args.audit.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
