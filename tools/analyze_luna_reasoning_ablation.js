const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const dataRoot = path.join(root, 'data', 'luna_hard_ablation_100');
const modes = ['light', 'medium', 'high', 'xhigh'];
const source = fs.readFileSync(path.join(dataRoot, 'ablation_100.jsonl'), 'utf8')
  .split(/\r?\n/).filter(Boolean).map(JSON.parse);

function loadOutcomes(mode) {
  const dir = path.join(dataRoot, 'runs', `luna_${mode}`);
  const result = {};
  for (const [filename, status] of [
    ['passed.jsonl', 'pass'],
    ['needs_audit.jsonl', 'audit'],
    ['exhausted_normal.jsonl', 'exhausted'],
    ['tail_escalation.jsonl', 'tail'],
  ]) {
    const file = path.join(dir, filename);
    if (!fs.existsSync(file)) continue;
    for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean)) {
      const row = JSON.parse(line);
      result[String(row.record_uid)] = { status, row };
    }
  }
  return result;
}

function loadRaw(mode) {
  const file = path.join(dataRoot, 'runs', `luna_${mode}`, 'raw_batches.jsonl');
  return fs.existsSync(file)
    ? fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse)
    : [];
}

const outcomes = Object.fromEntries(modes.map(mode => [mode, loadOutcomes(mode)]));
const valid = status => status === 'pass' || status === 'audit';
const summary = {};

for (const mode of modes) {
  const map = outcomes[mode];
  const counts = { pass: 0, audit: 0, exhausted: 0, tail: 0, missing: 0 };
  for (const row of source) counts[map[row.record_uid]?.status || 'missing'] += 1;
  summary[mode] = {
    ...counts,
    terminal: 100 - counts.missing,
    valid_candidates: counts.pass + counts.audit,
    valid_rate_all_100: (counts.pass + counts.audit) / 100,
    valid_rate_terminal: (counts.pass + counts.audit) / Math.max(1, 100 - counts.missing),
    by_route: {},
    by_bucket: {},
  };
  for (const field of ['route', 'length_bucket']) {
    const target = field === 'route' ? summary[mode].by_route : summary[mode].by_bucket;
    for (const value of [...new Set(source.map(row => row[field]))]) {
      const rows = source.filter(row => row[field] === value);
      const c = { n: rows.length, pass: 0, audit: 0, exhausted: 0, tail: 0, missing: 0 };
      for (const row of rows) c[map[row.record_uid]?.status || 'missing'] += 1;
      c.valid_candidates = c.pass + c.audit;
      target[value] = c;
    }
  }
}

const commonLMH = source.filter(row => ['light', 'medium', 'high'].every(mode => outcomes[mode][row.record_uid]));
const commonCohort = { n: commonLMH.length };
for (const mode of ['light', 'medium', 'high']) {
  const c = { pass: 0, audit: 0, valid_candidates: 0 };
  for (const row of commonLMH) {
    const status = outcomes[mode][row.record_uid].status;
    c.pass += status === 'pass';
    c.audit += status === 'audit';
    c.valid_candidates += valid(status);
  }
  c.valid_rate = c.valid_candidates / commonLMH.length;
  commonCohort[mode] = c;
}

const paired = {};
for (const [lower, higher] of [['light', 'medium'], ['medium', 'high'], ['light', 'high']]) {
  const common = source.filter(row => outcomes[lower][row.record_uid] && outcomes[higher][row.record_uid]);
  const improved = [], worsened = [];
  for (const row of common) {
    const low = valid(outcomes[lower][row.record_uid].status);
    const high = valid(outcomes[higher][row.record_uid].status);
    if (!low && high) improved.push(row.record_uid);
    if (low && !high) worsened.push(row.record_uid);
  }
  paired[`${lower}_to_${higher}`] = { common: common.length, improved, worsened };
}

const timeout = {};
for (const mode of modes) {
  timeout[mode] = {};
  for (const row of loadRaw(mode)) {
    const run = timeout[mode][row.run_id] ||= { requests: 0, timed_out: 0, infrastructure_failure: 0, no_final: 0, elapsed_seconds: 0, credits: 0 };
    run.requests += 1;
    run.timed_out += Boolean(row.timed_out);
    run.infrastructure_failure += Boolean(row.infrastructure_failure);
    run.no_final += row.raw_final == null;
    run.elapsed_seconds += Number(row.elapsed_seconds || 0);
    run.credits += Number(row.estimated_credits || 0);
  }
  for (const run of Object.values(timeout[mode])) {
    run.average_elapsed_seconds = run.elapsed_seconds / run.requests;
    run.timeout_rate = run.timed_out / run.requests;
    run.credits = Number(run.credits.toFixed(6));
  }
}

const result = {
  generated_at: new Date().toISOString(),
  source_records: source.length,
  summary,
  common_cohort: commonCohort,
  paired_transitions: paired,
  timeout,
  notes: {
    valid_candidate: 'pass or audit; leetspeak candidates are always routed to audit by policy',
    missing: 'no terminal queue row; mostly timeout/deferred for Light/Medium/High and pending for XHigh',
  },
};

const output = path.join(dataRoot, 'luna_reasoning_ablation_analysis.json');
fs.writeFileSync(output, JSON.stringify(result, null, 2), 'utf8');
console.log(output);
