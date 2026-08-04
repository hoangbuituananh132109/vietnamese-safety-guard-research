# CV Project Brief — Nemotron Safety Guard v3 EN→VI

> Bản tóm tắt dùng làm nguồn sự thật để viết CV, portfolio hoặc project description. Phạm vi của bản này gồm pipeline dịch chính bằng Gemini, 7 run encoder chính và nhánh nghiên cứu decoder Qwen3Guard base/fine-tune đã hoàn tất trên corpus đó. Các nghiên cứu Luna/Sol mới hoàn thành ở giai đoạn dịch, chưa được dùng để huấn luyện/đánh giá, nên không được tính là kết quả nghiên cứu của dự án.

## Tên dự án mong muốn

**Nemotron Safety Guard v3 EN→VI: Evidence-first translation pipeline and bilingual safety-guard experiments**

Tên ngắn cho CV:

**English–Vietnamese Safety Dataset Translation and Bilingual Guard Evaluation**

## Thời gian thực hiện

**07/2026–07/2026 cho bản dịch và QA chính; 07/2026 cho ma trận thí nghiệm Phase 0.**

Nếu cần ghi rộng hơn theo lịch sử artifact: **07/2026–08/2026**, nhưng phần kết quả được trình bày trong CV này chỉ dùng dữ liệu Gemini và các run thí nghiệm đã khóa trong tháng 07/2026.

## Tóm tắt CV-ready

Built an evidence-first English-to-Vietnamese translation pipeline for the NVIDIA Nemotron Safety Guard v3 corpus, covering 45,416 safety records while preserving JSON structure, labels, URLs, code, placeholders, null/empty semantics, profanity, slurs, harmful intent, and leetspeak. Implemented checkpointed structured-output translation, UID-based joining, recursive batch splitting, hard validators, provenance tracking, revision queues, and bilingual/manual QA. Used the resulting Gemini-based EN–VI corpus to run controlled GLiGuard, mmBERT and Qwen3Guard experiments, measuring bilingual gains, full-data scaling, dynamic-schema behavior, decoder zero-shot quality, LoRA fine-tuning gains, cross-lingual performance, N23 multi-label quality, and external SEA-SafeguardBench generalization.

## Dữ liệu nguồn

- **Tên chính xác:** NVIDIA Nemotron Safety Guard Dataset v3 / `Nemotron Safety Guard v3`.
- **Phiên bản:** v3; giữ nguyên official train/validation/test split của nguồn.
- **Đường dẫn nguồn cục bộ:** `data/raw/nemotron_safety_guard_v3/en/{train,valid,test}.jsonl`.
- **Tổng số mẫu nguồn:** **45.416** record.
  - Train: 40.007
  - Validation: 2.445
  - Test: 2.964
- **Số mẫu tiếng Việt cuối cùng:** **45.416/45.416, độ phủ 100%**.
- **Trường văn bản được dịch:** `prompt` và `response` khi có nội dung; giữ nguyên bản tiếng Anh trong `prompt_en`/`response_en` và thêm `prompt_vi`/`response_vi`.
- **Nhãn chính:** `safe`/`unsafe` cho prompt và response.
- **Cấu trúc safety:** 23 atomic safety categories, 1.669 tổ hợp nhãn; một record có thể có nhiều category.
- **Tag dữ liệu:** `generic` và `jailbreaking` đều được giữ lại; không loại mẫu chỉ vì có tag jailbreak.
- **Các nhóm nội dung:** criminal planning/confessions, hate/identity hate, violence, harassment, controlled substances, PII/privacy, profanity, sexual content, self-harm, illegal activity, weapons, malware, fraud/deception, threat, manipulation, misinformation/conspiracy và các nhóm safety liên quan khác.

## Pipeline: từ tiếng Anh đến tiếng Việt

1. Audit source JSONL trước khi dịch: parse JSON, đếm split, kiểm tra duplicate, missingness, null/empty response, label states, category distribution, độ dài và cross-split overlap.
2. Tạo khóa ổn định `record_uid` từ split, số dòng vật lý và SHA-256 của source line; không dùng `id` của NVIDIA làm khóa duy nhất vì một ID có thể xuất hiện ở nhiều record augmentation.
3. Chọn pilot và holdout có kiểm soát để phát hiện lỗi profanity, thuật ngữ, JSON/schema, leetspeak, nội dung nguy hại và bản dịch bị làm nhẹ trước khi scale.
4. Dịch `prompt`/`response` bằng structured output của Gemini, giữ nguyên source fields và gắn metadata/provenance cho từng record.
5. Chạy checkpoint/resume theo từng record; flush/fsync kết quả đã hoàn thành, ghi failure queue, vô hiệu hóa API-key slot bị provider từ chối và tiếp tục bằng slot khác.
6. Dùng batching theo P95/length bucket; khi structured output hoặc semantic validation lỗi, chia batch đệ quy đến singleton để cô lập record hỏng thay vì làm dừng toàn bộ job.
7. Chạy validator cấu trúc và hard-quality validator; các record không đạt được đưa vào revision queue, không bị âm thầm bỏ đi.
8. Sửa/retranslate có provenance bằng Gemini revision, Terra/manual review và Codex-assisted human review khi cần; lưu hash trước/sau, lý do sửa và trạng thái audit.
9. Materialize final JSONL và kiểm tra parity theo UID, hash, số ký tự, split, null/empty semantics, duplicate/extra/missing record và trainer compatibility.

### Mô hình/API dùng để dịch

- **Provider chính:** Gemini structured-output translation.
- **Model chính trong artifact:** `gemini-3.1-flash-lite`.
- **Revision/review:** Gemini revision, Terra review, web/manual bilingual review và Codex-assisted human review cho các hàng bị cảnh báo hoặc cần sửa.
- **Fallback có dùng không?** Có fallback ở cấp pipeline/revision, nhưng mọi provider/stage đều được ghi rõ trong provenance. Đây không phải silent fallback; output cuối giữ `translation_provider`, `translation_model`, `translation_prompt_version`, `translation_batch_id`, attempt và hash.

### Cách phát hiện bản dịch lỗi

- Parse/JSON schema và kiểm tra đúng field.
- So khớp `record_uid`, source hash, translation hash và input/output character counts.
- Kiểm tra preservation của key JSON, URL, identifier, placeholder, timestamp, code block, literal token và null/empty response.
- Phát hiện copied English, untranslated English/leet, leetspeak chưa được xử lý, profanity bị làm nhẹ, identity slur bị bỏ nguyên văn, semantic drift và output bị cắt.
- Kiểm tra bilingual/structural consistency, language signal, length bucket và hard-tail routing.
- Heuristic chỉ là audit cue khi gặp code, SQL/JSON, URL, tên riêng hoặc văn bản đa ngôn ngữ; không tự động coi mọi similarity hoặc fenced code là bản dịch lỗi.

### Validator và cách sửa record không đạt

- `translator.validators` và các hard validator của pipeline.
- JSON/structural validator, schema/key validator, URL/placeholder/token preservation checks.
- Profanity, slur, leetspeak, copied-English và semantic hard-quality checks.
- UID/hash/coverage/integrity audit và final materialization audit.
- Record lỗi được retry/retranslate, tách batch, đưa vào queue, sửa theo review route rồi chạy validator lại. Record được sửa không bị ghi đè mất lịch sử; trạng thái, provider, rationale và hash trước/sau được giữ lại.

### Kiểm tra thủ công

- Pilot holdout 100 record: **95 approved, 5 minor-fix, 0 critical; 100/100 đạt ít nhất 8/10**.
- Final difficult/revision queue: **478/478 hoàn tất, còn 0 failure queue**.
- Final quality report: **45.330 pass trực tiếp + 86 human-audited overrides; 0 hard-validator error không được giải thích**.
- 64 record từng bị cảnh báo profanity được bilingual-review; record bị làm nhẹ được sửa, cảnh báo giả được approve kèm audit trail.

## Bảo toàn dữ liệu và fidelity safety

- Không sửa `data/raw`; giữ source English và thêm bản dịch song song.
- Giữ nguyên JSON key/schema, labels, tags, IDs, `record_uid`, source line metadata, URLs, identifiers, timestamps, placeholders, code, SQL/JSON literals và formatting cần thiết.
- Phân biệt `response = null` với `response = ""`; không dịch empty response thành nội dung và không biến null thành empty string.
- Không làm nhẹ nội dung nguy hại, slur, profanity, sexual content, self-harm, threats, criminal intent hoặc jailbreak prose. Mục tiêu là tương đương về ý nghĩa, mức độ trực tiếp, register, aggression, target identity và intent.
- Với leetspeak: phát hiện cả obfuscation và copied English; dịch meaning trước, sau đó tái tạo mức obfuscation phù hợp khi chính sách yêu cầu. Các literal token quan trọng vẫn phải được giữ và audit.
- Không dịch tên nhãn safety/provenance như thể đó là nội dung tự nhiên; các field metadata được giữ nguyên để không làm hỏng khả năng tái lập.

## 7 thí nghiệm chính đã hoàn thành

> Cách đếm 7 run theo ma trận khóa của dự án: **B0 + E1 + E2 + E3 + E4 + E5 + E7**. E6 binary-only no-R là một ablation phụ, không nằm trong ma trận chính và không được dùng để thay thế một run trong 7 mục dưới đây. Các headline của 7 run trong phần này là evaluation contract full-R/full suite; báo cáo no-R follow-up có population khác và được tách riêng trong [phụ lục bằng chứng chi tiết](<D:\Downloads\Safety Dataset\reports\CV_DETAILED_EXPERIMENT_EVIDENCE_NEMOTRON_EN_VI.md>).

### Thí nghiệm 1 — B0-GLI-ZS: GLiGuard zero-shot baseline

- **Mô hình:** released GLiGuard, không fine-tune.
- **Tập train:** không có.
- **Tập đánh giá:** native/shared GLi-compatible subset; Nemotron test 8.476 instance và SEA paired 3.146 instance.
- **Baseline:** sanity check random/majority theo experiment contract; con số chính dùng B0 làm zero-shot reference.
- **Metric:** binary safety accuracy; phân tích thêm theo EN/VI, P/PR, unsafe recall và calibration.
- **Kết quả:** Nemotron **66,80%**, SEA **68,88%**.
- **Ý nghĩa:** tạo mốc zero-shot để đo lợi ích của fine-tuning GLiGuard và thay backbone.

### Thí nghiệm 2 — E1-G-EV-512: GLiGuard + LoRA

- **Mô hình:** GLiGuard + LoRA, native 512-token contract.
- **Tập train:** các cặp EN–VI GLi-native; 57.804 semantic pairs, 115.608 instances theo manifest khóa.
- **Tập đánh giá:** cùng primary native subset với B0; 4.238 Nemotron test pairs và 1.573 SEA pairs.
- **Baseline:** B0 zero-shot trên đúng miền đánh giá.
- **Metric:** accuracy, macro-F1, unsafe precision/recall/F1, AUPRC/AUROC, confusion matrix và EN–VI agreement.
- **Kết quả:** Nemotron **68,70%**; SEA **65,00%**.
- **So với baseline:** Nemotron tăng khoảng **+1,90 điểm phần trăm**; SEA giảm khoảng **−3,88 điểm phần trăm**, cho thấy fine-tuning GLiGuard không tự động cải thiện external cross-lingual generalization.

### Thí nghiệm 3 — E2-M-EV-GLI-COMPAT-8K: mmBERT fixed-head trên cùng cặp

- **Mô hình:** mmBERT fixed-head + LoRA, được phép dùng đến 8K token.
- **Tập train:** đúng cùng ID/text với E1, để kiểm soát khác biệt data membership; 115.608 instances theo GLi-compatible manifest.
- **Tập đánh giá:** cùng native/shared subset với E1.
- **Baseline:** E1 GLiGuard + LoRA và B0 trên cùng row-matched IDs.
- **Metric:** binary accuracy/macro-F1/AUPRC, unsafe recall, EN–VI agreement, native-vs-truncated audit.
- **Kết quả:** Nemotron **77,18%**; SEA **72,98%**.
- **So với E1:** tăng **+8,48 điểm** trên Nemotron và **+7,98 điểm** trên SEA.
- **Ý nghĩa:** trên cùng dữ liệu và cùng cặp đánh giá, mmBERT fixed-head mạnh hơn GLiGuard recipe 512 hiện tại.

### Thí nghiệm 4 — E3-M-E-8K: English-only control

- **Mô hình:** mmBERT fixed-head + LoRA.
- **Tập train:** English-only, 70.068 train instances theo ngân sách matched.
- **Tập đánh giá:** full 8K suite; Nemotron test 10.682 instances và SEA paired 3.680 instances; đánh giá riêng EN/VI.
- **Baseline:** English-only control để đo tác động của việc bổ sung tiếng Việt.
- **Metric:** accuracy, macro-F1, unsafe recall, AUPRC, EN–VI gap, paired decision consistency; N23 micro/macro-F1 trên 5.768 hàng có supervision.
- **Kết quả:** Nemotron **75,58%**, SEA **71,82%**; N23 micro-F1 **0,2373**, macro-F1 **0,0826**.
- **Ý nghĩa:** đây là mốc “chưa bổ sung dữ liệu tiếng Việt”.

### Thí nghiệm 5 — E4-M-EV-MATCHED-8K: matched bilingual data

- **Mô hình:** cùng mmBERT fixed-head + LoRA và recipe của E3.
- **Tập train:** 70.068 semantic units, chọn một ngôn ngữ ổn định theo `record_uid` hash; 35.248 EN và 34.820 VI.
- **Tập đánh giá:** cùng full 8K Nemotron/SEA suite như E3.
- **Baseline:** E3 English-only với cùng ngân sách train.
- **Metric:** binary metrics như E3; N23 micro/macro-F1, exact match và EN–VI gap.
- **Kết quả:** Nemotron **76,13%**, SEA **72,58%**; N23 micro-F1 **0,2366**, macro-F1 **0,0702**.
- **So với E3:** Nemotron tăng **+0,55 điểm**, SEA tăng **+0,76 điểm**; trên slice VI, Nemotron tăng từ **73,17% lên 74,80%** và SEA từ **69,84% lên 71,09%**.
- **Ý nghĩa:** exposure tiếng Việt giúp cải thiện khả năng xử lý VI và thu hẹp language gap khi ngân sách được kiểm soát.

### Thí nghiệm 6 — E5-M-EV-FULL-8K: full bilingual data

- **Mô hình:** mmBERT fixed-head + LoRA.
- **Tập train:** toàn bộ EN+VI, 140.136 train instances.
- **Tập đánh giá:** full 8K Nemotron test 10.682 và SEA paired 3.680; N23 trên 5.768 supervised rows.
- **Baseline:** E4 matched bilingual, đồng thời so sánh với E3 English-only.
- **Metric:** binary accuracy/macro-F1/unsafe recall/AUPRC, EN–VI gap/agreement, N23 micro/macro-F1/exact match.
- **Kết quả:** Nemotron **78,58%**, SEA **74,13%**; N23 micro-F1 **0,3931**, macro-F1 **0,2042**, exact match **46,41%**.
- **So với E4:** Nemotron tăng **+2,45 điểm**, SEA tăng **+1,55 điểm**. So với E3, full EN+VI tăng **+3,00 điểm** trên Nemotron và **+2,31 điểm** trên SEA.
- **Kết luận:** E5 là fixed-head encoder tốt nhất trong ma trận; quy mô full bilingual có lợi cho cả EN lẫn VI.

### Thí nghiệm 7 — E7-M-SCHEMA-EV-8K: dynamic schema GLi-style

- **Mô hình:** mmBERT dynamic-schema GLi-style + LoRA, marker embeddings và shared scalar MLP cho binary/N23.
- **Tập train:** cùng full EN+VI 140.136 instances như E5 để giữ data budget.
- **Tập đánh giá:** cùng full 8K Nemotron/SEA suite và N23 supervised rows.
- **Baseline:** E5 fixed-head trên cùng full EN+VI data; B0/E1/E2 là architecture references.
- **Metric:** binary accuracy/macro-F1/AUPRC/unsafe recall, canonical-vs-reversed schema consistency, N23 micro/macro-F1 và calibration.
- **Kết quả:** Nemotron **56,13%**, SEA **54,54%**; N23 micro-F1 **0,0000**, macro-F1 **0,0000** tại threshold 0,5; accuracy E5 tương ứng **78,58%/74,13%**.
- **Trước/sau:** dynamic schema thấp hơn E5 **−22,45 điểm** trên Nemotron và **−19,59 điểm** trên SEA.
- **Chẩn đoán:** run hoàn tất đủ step, không NaN, không thiếu gradient hay OOM; suy giảm là kết quả model/optimization của recipe 2 epoch + rank-4 LoRA hiện tại, không phải lỗi pipeline chạy dở.

## Nghiên cứu bổ sung đã hoàn thành — Qwen3Guard base và fine-tune

> Hai run dưới đây là nhánh decoder bổ sung, không thay thế 7 run encoder chính ở trên. Qwen có taxonomy native khác Nemotron N23, vì vậy kết quả được chấm trên binary Safe/Unsafe dùng chung; không gán nhãn Qwen vào 23-category N23.

### Q1 — Qwen3Guard-Gen-4B base, zero-shot no-R

- **Mô hình:** `Qwen/Qwen3Guard-Gen-4B`, full weights, không quantization; pinned revision `6ec42827da0c1ff11e7a49dc269d2e810d27e108`; inference bằng vLLM FP16, native tokenizer chat template.
- **Tập train:** không có; zero-shot.
- **Tập đánh giá:** cùng một manifest no-R gồm **11.736** mẫu P/PR: Nemotron test **8.056** và SEA paired **3.680**, cân bằng EN **5.868** và VI **5.868**; không đánh giá response-only R.
- **Baseline:** zero-shot decoder baseline; các số E3/E4/E5/E7 và D1 được đối chiếu trên cùng exact-ID/no-R contract khi có thể.
- **Metric:** binary accuracy và Macro-F1; hai policy được báo riêng. Policy conservative map `Controversial → Unsafe`; policy lenient map `Controversial → Safe`. N23 không chấm vì taxonomy không tương đương.
- **Kết quả chính (conservative):** **85,15% overall**, **85,86% Nemotron**, **83,59% SEA**, Macro-F1 **0,8494**; Unsafe recall **93,88%**, Safe recall **75,85%**.
- **Kết quả policy đối chiếu (lenient):** **84,91% overall**, **85,22% Nemotron**, **84,24% SEA**, Macro-F1 **0,8490**.
- **Phát hiện:** model có xu hướng bắt Unsafe tốt hơn nhưng false-block Safe nhiều hơn dưới conservative mapping; EN–VI accuracy gap là **1,82 điểm** và paired decision agreement là **94,29%**.

### Q2 — Qwen3Guard-Gen-4B + LoRA, binary no-R fine-tune

- **Mô hình:** `Qwen/Qwen3Guard-Gen-4B` + LoRA adapter; BF16, LoRA rank **8**, alpha **32**, dropout **0,05**, target `q_proj`/`v_proj`, effective batch **32**, max sequence length **2.048**.
- **Tập train:** full no-R EN+VI manifest **101.274** mẫu từ corpus Gemini đã materialize; **1 epoch**, **3.165 optimizer steps**; target được khóa là một dòng binary `Safety: Safe/Unsafe`, không giả định taxonomy native của Qwen là N23.
- **Tập đánh giá:** giữ nguyên manifest của Q1: **11.736** mẫu P/PR Nemotron test + SEA, gồm **5.868 EN** và **5.868 VI**; R bị loại trước khi load model.
- **Baseline:** Q1 Qwen base zero-shot trên cùng evaluation contract.
- **Metric:** strict binary parsing, accuracy, Macro-F1, Safe/Unsafe precision-recall-F1, confusion matrix, Nemotron/SEA slice và EN–VI paired agreement; N23 không chấm.
- **Kết quả sau fine-tune:** **87,83% overall**, **88,49% Nemotron**, **86,39% SEA**, Macro-F1 **0,8782**; EN **88,45%**, VI **87,22%**, paired EN–VI same-decision rate **93,83%**; **11.736/11.736** output parsed strict, **0** truncation và **0** parse failure.
- **Trước → sau so với Q1 conservative:** overall **85,15% → 87,83% (+2,68 điểm)**; Nemotron **85,86% → 88,49% (+2,63 điểm)**; SEA **83,59% → 86,39% (+2,80 điểm)**.
- **Ý nghĩa:** fine-tuning binary trên full EN+VI cải thiện rõ cả benchmark cùng miền và SEA; đây là kết quả decoder/LoRA bổ sung, không phải bằng chứng Qwen đã học hoặc tái tạo được N23 multi-label.

## Kết quả tổng hợp và diễn giải

- **Bilingual data:** E3 → E4 → E5 cho thấy bổ sung tiếng Việt và tăng từ matched lên full bilingual đều cải thiện kết quả; full EN+VI là lựa chọn tốt nhất trong fixed-head matrix.
- **Cross-lingual:** E5 đạt 77,20% trên VI Nemotron và 73,04% trên VI SEA; vẫn còn language gap nên không được mô tả là đã giải quyết hoàn toàn Vietnamese safety.
- **Architecture:** E2 mmBERT fixed-head vượt E1 GLiGuard trên cùng subset; E7 dynamic schema chưa cạnh tranh trong recipe hiện tại.
- **Decoder:** Q1 cho thấy Qwen3Guard base zero-shot đã là một binary decoder mạnh; Q2 LoRA trên full EN+VI tăng **+2,68 điểm overall**, **+2,63 điểm Nemotron** và **+2,80 điểm SEA** so với Q1 trên cùng no-R evaluation contract.
- **N23:** E5 là run encoder tốt nhất với micro-F1 0,3931; E7 có dấu hiệu collapse calibration/learning ở threshold 0,5 và không nên được mô tả là chỉ cần đổi threshold là sẽ giải quyết.
- **External benchmark:** SEA-SafeguardBench là bằng chứng quan trọng hơn cho generalization. Cần báo riêng P/PR; overall accuracy có thể che giấu unsafe recall thấp ở PR.

## Báo cáo cũ và lớp đánh giá chi tiết

Các con số phần trăm trong bản CV chỉ là headline. Phân tích theo từng phần vẫn được giữ trong các báo cáo cũ và trong [Detailed Experiment Evidence](<D:\Downloads\Safety Dataset\reports\CV_DETAILED_EXPERIMENT_EVIDENCE_NEMOTRON_EN_VI.md>):

- [Tổng kết cuối encoder/Qwen/Nemotron](<D:\Downloads\Safety Dataset\reports\research_archive\PROJECT_FINAL_SYNTHESIS_ENCODERS_QWEN_NEMOTRON_PUBLISHED_20260724.md>): common leaderboard, Q1→Q2, D1/D2/D3, N23, local-vs-published metric và claim boundaries.
- [Q2 so với Q1/D1/D2](<D:\Downloads\Safety Dataset\reports\research_archive\Q2_VS_Q1_D1_D2_NO_R_20260724.md>): confusion matrix, EN/VI, Nemotron/SEA, P/PR, topic slices, paired correctness và McNemar.
- [Q2 so với Q1/D2/D3](<D:\Downloads\Safety Dataset\reports\research_archive\Q2_VS_Q1_D2_NO_R_20260724.md>): D3 Nemotron full-VI LoRA, binary slices và N23 theo view.
- [D3 Nemotron no-R](<D:\Downloads\Safety Dataset\reports\research_archive\D3_NEMOTRON_NO_R_FINAL_ANALYSIS_20260724.md>): training/evaluation contract, binary trade-off và N23 D1/D2/D3.
- [R versus no-R, E6 và E7](<D:\Downloads\Safety Dataset\reports\analysis_20260724\R_VS_NO_R_E6_E7_COMPREHENSIVE_ANALYSIS_20260724.md>): per-view unsafe recall, AUPRC, ECE, N23 theo label, E5/E6 ablation và calibration caveats.
- [Experiment synthesis cũ](<D:\Downloads\Safety Dataset\reports\research_archive\PROJECT_EXPERIMENT_SYNTHESIS_20260722.md>): shared task contract, metric definitions, error counts và runtime trade-offs.
- [Báo cáo chất lượng dịch cuối](<D:\Downloads\Safety Dataset\reports\final_quality\TRANSLATION_QUALITY_REPORT.md>): coverage, UID/hash integrity, validator, review queue, split và quality status theo safety label.

Các báo cáo này giải thích vì sao một kết luận đúng phải nói rõ **benchmark nào, ngôn ngữ nào, view P/PR/R nào, binary hay N23, threshold nào và có/không có R**; không nên chỉ đưa một accuracy tổng.

### Decoder reference ngoài 7 run chính

D1 — `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`, full FP16/vLLM — đạt **86,86% Nemotron test** và **83,91% SEA**, cao hơn E5 lần lượt **+8,28** và **+9,78 điểm**. Đây là reference decoder, không tính vào 7 run B0/E1–E5/E7. So sánh cần ghi caveat: D1 được phát triển trên cùng model family/dataset family và có lợi thế domain ở Nemotron; SEA là bằng chứng độc lập hơn.

## Đóng góp cá nhân — bản nháp CV

Các bullet dưới đây có thể dùng ở ngôi thứ nhất hoặc đổi thành động từ CV:

- Designed and implemented an evidence-first EN→VI safety-data translation pipeline for 45.4K Nemotron records, with structured output, checkpoint/resume, multi-key rotation, recursive batch splitting and failure-queue recovery.
- Built UID-, hash- and provenance-based data integrity checks that preserved official splits and prevented position-based EN/VI misalignment, duplicate rows, missing records and silent provider mixing.
- Developed hard validators for JSON/schema fidelity, URLs, placeholders, identifiers, code, null/empty semantics, profanity, identity slurs, harmful intent, copied English and leetspeak.
- Combined automated validation with targeted bilingual/manual review; completed the difficult queue and recorded repair/override decisions instead of hiding heuristic failures.
- Designed a controlled B0/E1–E5/E7 experiment matrix separating zero-shot baseline, GLiGuard vs mmBERT, English-only vs matched bilingual vs full bilingual training, and fixed-head vs dynamic-schema modeling.
- Ran and analyzed GPU training/evaluation artifacts with row-matched Nemotron and SEA benchmarks, language slices, N23 multi-label metrics, paired EN–VI consistency and calibration/error analysis.
- Evaluated Qwen3Guard-Gen-4B both zero-shot and after a reproducible binary Safe/Unsafe LoRA fine-tune, using a shared 11,736-example no-R contract and a 101,274-example full EN+VI training manifest; quantified the before/after gains on Nemotron and SEA.
- Reported negative results honestly: the current dynamic-schema recipe completed technically but underperformed, while full bilingual fixed-head training was the strongest encoder condition.

## Tech stack

Python, PowerShell, JSONL streaming, Gemini structured output/API, checkpoint/resume pipelines, recursive batching, custom validators, SHA-256 provenance, PyTorch, Hugging Face Transformers, PEFT/LoRA, mmBERT, GLiGuard/GLiNER-family components, Qwen3Guard-Gen-4B, vLLM, CUDA GPU, pytest, Jupyter/Notebook, SEA-HELM/SEA-SafeguardBench, offline HTML/JSON/Markdown reporting.

## Links và artifact

- **GitHub:** chưa có remote/repository public được xác nhận trong workspace hiện tại.
- **Hugging Face model reference:** [nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3)
- **Qwen model reference:** [Qwen/Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B)
- **Dataset HF:** chưa có link public cho bản EN–VI của dự án.
- **Báo cáo dịch:** `reports/final_quality/TRANSLATION_QUALITY_REPORT.md`
- **Machine-readable translation audit:** `reports/final_quality/translation_quality_summary.json`
- **English source audit:** `reports/nemotron_safety_audit.md`
- **Experiment contract:** `GUARD_EXPERIMENT_EXECUTION_PLAN_LOCKED_V3.md`
- **E7 analysis:** `reports/phase0/E7_SCHEMA_ANALYSIS.md`
- **Qwen base analysis:** `reports/research_archive/QWEN3GUARD_Q1_ANALYSIS_20260723.md`
- **Qwen fine-tune run contract:** `reports/vast_download/d3_nemotron_no_r_4080s_20260724/extracted/results/no_r_decoder_4080s/train/qwen/run_contract.json`
- **Qwen fine-tune metrics:** `reports/vast_download/d3_nemotron_no_r_4080s_20260724/extracted/results/no_r_decoder_4080s/eval/qwen/metrics.json`
- **Cross-experiment analysis:** `reports/decoder_baseline/D1_E7_FINAL_ANALYSIS.md`
- **Main R/no-R/E6/E7 analysis:** `reports/analysis_20260724/R_VS_NO_R_E6_E7_COMPREHENSIVE_ANALYSIS_20260724.md`
- **Detailed CV evidence appendix:** `reports/CV_DETAILED_EXPERIMENT_EVIDENCE_NEMOTRON_EN_VI.md`
- **Core code:** `translator/`, `tools/`, `guard_train/`, `guard_smoke/`, `tests/`

## Ranh giới claim khi đưa vào CV

- Nói **Gemini là pipeline dịch chính đã hoàn tất**, không nói Luna/Sol đã tạo ra kết quả train/eval.
- Không claim bản dịch đạt BLEU/COMET hoặc human-perfect quality trên toàn bộ 45.416 record; audit hiện tại chứng minh coverage, structure, provenance, validators và review completion.
- Không claim dynamic-schema GLi-style đã thành công; kết quả hiện tại là negative/diagnostic result.
- Không claim bỏ response luôn tốt hơn; các ablation response/no-response bị confound bởi số lượng supervision và cần báo caveat.
- Không claim SEA là hoàn toàn độc lập với Nemotron ở mọi subset; General có overlap/provenance caveat, trong khi ITW/CG là bằng chứng văn hóa hữu ích hơn.
- Với Qwen, chỉ claim binary Safe/Unsafe trên no-R P/PR contract; không claim Qwen base/fine-tune đã đạt hoặc tái tạo taxonomy N23.
- Nếu đưa E6 vào portfolio, ghi là **auxiliary fixed-head binary-only no-R ablation**, không thay thế run thứ 7 của ma trận chính.
