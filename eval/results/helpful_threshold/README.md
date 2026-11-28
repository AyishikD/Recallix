# helpful mode: a lower answer threshold (2026-11-28, development data)

Recall's reader (deepset/roberta-base-squad2) declines many clear questions in texting-style memories because its
confidence sits just under the threshold (-2.721). A sweep of lower thresholds for every recall (update scenarios
gemini_upd1-3, 90; Gemini retrieval batches, 30 traps) showed each step answering more and also answering more traps:

| threshold | updates correct / wrong / declined | Gemini traps declined (of 30) | Gemini false declines |
|---|---|---|---|
| -2.72 (default) | 57 / 5 / 16 | 28 | 0.12 |
| -3.5 | 61 / 5 / 13 | 27 | 0.12 |
| -4.5 | 63 / 5 / 11 | 26 | 0.093 |
| -6 | 64 / 6 / 8 | 26 | 0.08 |

So the default is unchanged; the lower threshold is opt-in, in helpful mode: `recall.helpful_threshold_offset: 0.8`
(about -3.5). helpful, offset 0 vs 0.8 (files offset_<o>_*.json):

| | updates (90) correct / wrong / declined | Gemini traps declined | LoCoMo dev real wrongly declined (of 760) | LoCoMo dev traps declined (of 237) | LoCoMo answer F1 | dev set |
|---|---|---|---|---|---|---|
| offset 0 | 57 / 5 / 16 | 28 / 30 | 56 | 77 | 0.3758 | unchanged |
| offset 0.8 | 61 / 5 / 13 | 27 / 30 | 45 | 68 | 0.3753 | unchanged |

On LoCoMo the trade is almost one trick answer per extra real answer (11 vs 9), which is why it stays out of balanced.
All data here is development data. A better reader for texting-style text, not a threshold, is what would answer more
without answering more traps.
