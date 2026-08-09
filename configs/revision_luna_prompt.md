You are reviewing an English-to-Vietnamese safety-dataset translation.

Use the English source as authoritative. Compare all supplied Vietnamese candidates and produce one corrected Vietnamese record.

Requirements:
- Preserve meaning, intent, toxicity, profanity, slurs, threats, PII, obfuscation, and speaker voice. Do not sanitize.
- Preserve null versus empty string exactly.
- Preserve record_uid exactly.
- Preserve REDACTED markers, URLs, emails, code fences, JSON/schema keys, placeholders, and identifiers exactly.
- Fix only the listed validation problems plus any obvious local defect required for a coherent translation.
- Do not add refusals, safety advice, explanations, or facts not present in the source.
- Return only the required structured object: record_uid, prompt_vi, response_vi, review_status="revised", fix_notes.

Prefer the best existing candidate and make the smallest sufficient correction.
