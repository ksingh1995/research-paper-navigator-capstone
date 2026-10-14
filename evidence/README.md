# Recorded evidence

These files contain the saved capstone experiments, rather than newly rerun benchmarks. `python navigator.py summary` recalculates the key historical summaries. `python navigator.py replay` prints saved Week 7 wording unchanged.

Week 3 uses rubric grading plus deterministic retrieval, citation-presence, and abstention checks. Its 8/8 judge passes become 7/8 combined acceptances because the unsupported question fails the stricter guard.

Week 4 uses the six broad-direction questions requested for the final decomposition comparison. Both paired paths pass the same six grading notes. Additional files in the results do not establish higher factual accuracy.

Week 5 compares fixed retrieval, an initial agent, and a refined agent on three questions. Week 6 audits three model configurations on ordinary questions and two configurations on nine challenging cases. The probes were deliberately written for the audit, including synthetic overlays. They are not nine naturally occurring user incidents.

Week 7 records seven baseline/hardened pairs and one exact repeat. `superseded_records` preserves iterative runs but is excluded from final totals. S1 is a clear paired recovery from the clucking instruction. Baseline also resisted the visible fabricated fact and retrieved command in other pairs. The hash gate adds a deterministic boundary without proving a new answer-level rescue in those cases. A residual table-reading error remains in S3.

Only local path metadata is made portable. Generated answer wording remains unchanged. Exported log copies replace private workstation paths if present. Source logs and original course artifacts remain local and unchanged. Source PDFs are separate local inputs and are not published in the updated repository.

## File inventory

| Stage | Saved results | Questions and grading definitions | Answer/trace log |
| --- | --- | --- | --- |
| Week 2 baseline | `week2_baseline_results.json` | Four representative questions in the result records | `logs/checkpoint_2_1_retrieval.log` |
| Week 2 hybrid | `week2_results.json` | `test_sets/checkpoint_2_1_test_set.json` | `logs/checkpoint_2_1_hybrid_retrieval.log` |
| Week 3 evaluator | `week3_results.json` | `test_sets/checkpoint_3_1_test_set.json` | `logs/checkpoint_3_1_evaluation.log` |
| Week 4 decomposition | `week4_results.json` | `test_sets/checkpoint_4_1_broad_direction_test_set.json` | `logs/checkpoint_4_1_broad_direction.log` |
| Week 5 agents | `week5_results.json` | `test_sets/checkpoint_5_1_test_set.json` | `logs/checkpoint_5_1_agent.log` |
| Week 6 audit | `week6_results.json` and `reviews/checkpoint_6_1_*review*.json` | `test_sets/checkpoint_6_1_test_set.json` | `logs/checkpoint_6_1_agent.log` |
| Week 7 efficiency/guardrails | `week7_results.json` | `test_sets/checkpoint_7_1_test_set.json` | `week7_answers_verbatim.log` |
| Week 8 portable interface | `final_interface_method.json`, `final_interface_comparison_initial.json`, `final_interface_comparison.json`, `final_interface_repeat.json`, `final_keyword_check.json` | Questions are in each result record | Full answers and traces are in JSON, with `final_interface_review.txt` documenting the citation issue |

The Week 2 test-set export takes questions from saved hybrid results. Week 7 takes its case definitions from saved `test_cases`. Weeks 3–6 copy the recorded standalone test sets. Expected-source labels, title labels, and grading notes are evaluation data, not inputs to the final live navigator. Week 6 `control` has three ordinary cases, `security` has nine synthetic challenges, and `configs` records three model configurations. The two review JSON files are supporting judge audit records, not extra production runs or new measurements.

Week 4 evidence uses the six broad-direction questions. Superseded narrow-question Week 4 experiments, temporary smoke tests, caches, and worksheet-building scripts are deliberately outside this package. Week 7 retains superseded records *inside* its saved results/log and excludes them from reported final totals. The logs are execution records and can include multiple iterations or review blocks. The result JSON distinguishes final records, controls, and superseded runs.

## Reviewing and running

```sh
python scripts/verify_evidence.py
python navigator.py summary
python navigator.py replay --case P1 --arm hardened
python scripts/run_questions.py --test-set evidence/test_sets/checkpoint_6_1_test_set.json --group control
```

The question runner previews by default. `--execute` runs the current final system and saves new answers under ignored `outputs/`; it does not reproduce historical experimental arms or use an LLM judge. PDF-overlay cases require the original audit harness and the portable runner refuses them. Some user-input probes can be asked as ordinary questions, but that alone does not recreate an audit. No new API run is claimed by these exports or integrity checks.

Generated research answers and selected published-paper excerpts are intentionally retained for review. Do not put employer, patient, or credential data into these records. The repository scanner checks known token patterns and private home paths, not the factual accuracy or confidentiality of arbitrary prose.
