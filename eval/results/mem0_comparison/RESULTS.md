# Recallix vs Mem0 on LoCoMo: results (2026-11-24)

Protocol: eval/mem0_comparison/PROTOCOL.md (written before any run; amendments 1-4 logged before any answers
were seen).

- **Scope:** LoCoMo test conversation conv-44, 675 messages and 158 questions (123 answerable, 35 adversarial),
  under a $3 OpenAI cap.
- **Answering and judging:** identical for both systems (gpt-4o-mini, temperature 0).
- **Mem0:** mem0ai 2.2.1 with nlp and extras, gpt-4o-mini, text-embedding-3-small.
- **Recallix:** as shipped on 2026-11-24.

| | Recallix | Mem0 |
|---|---|---|
| Accuracy, all 158 questions | **67.7%** | 66.5% |
| Accuracy, 123 answerable | **61.0%** | 56.9% |
| Adversarial questions declined (35) | 91.4% | **100%** |
| Single-hop (62) | **79.0%** | 71.0% |
| Temporal (24) | **54.2%** | 41.7% |
| Multi-hop (30) | 43.3% | **53.3%** |
| Open-domain (7) | 0% | 0% |
| LLM calls to store 675 messages | **0** | 3,134 |
| Cost to store 675 messages | **$0** | $1.25 (≈ $0.0018 per message, estimated from token usage) |
| Store, median / p95 per message | **12 / 20 ms** | 2,489 / 3,807 ms |
| Recall, median / p95 per question | **142 / 197 ms** | 706 / 830 ms |

**Significance (paired, exact McNemar):**
- None of the accuracy differences is statistically significant on this sample.
- Overall: Recallix alone right on 22 questions, Mem0 alone right on 20, p = 0.88.
- By category, p is between 0.25 and 0.52.
- The defensible claim is **comparable accuracy**, not "better".

**Speed and cost:** the differences are not a matter of chance.
- Recallix stores without an LLM, so at $0 and about 200x faster per message.
- It recalls about 5x faster. Recallix was measured in-process on an Apple M4; Mem0's figures include OpenAI
  network calls.

**Caveats:**
- **One conversation.** conv-47 was stopped at 80 of 689 messages when Mem0's cost per message came in above
  calibration (amendment 4).
- **The answer model shapes the results.** Mem0's 100% on adversarial questions comes from gpt-4o-mini saying
  "I don't know" over Mem0's retrieved memories. Recallix declined 11 of 158 questions itself; on 3 adversarial
  questions its memories led the answer model to answer.
- **The judge is gpt-4o-mini,** which is lenient on dates, as in Mem0's published protocol.

**Total OpenAI spend:** $1.33 (estimated from token usage with a 10% margin). This includes the calibration and an
aborted first start (openai_spend_ledger.json).
