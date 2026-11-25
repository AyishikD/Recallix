# Retrieval fixes, 2026-11-24

What was tried after the head-to-head comparison showed most Recallix misses were retrieval misses
(31 of 36 "I don't know" answers on conv-44 never had the evidence in the top 10).

Development data only: LoCoMo dev (conversations 1-5, 999 questions), conv-44 (inspected question
by question for the comparison, so now development data), our dev set and the Gemini batches 1-3
(all development data by now). Measured with `measure.py` (evidence recall@10 of what recall
serves, declines, traps) and `eval/run_eval.py` (first-answer accuracy, traps). No LLM anywhere.

## Shipped

| Change | Setting | LoCoMo dev (before → after) | Other sets |
|---|---|---|---|
| Second pass: refill places 6-10 with a second search and the messages around the best result | `recall.second_pass` | served recall@10 0.649 → 0.706 | conv-44 0.606 → 0.667; dev, Gemini unchanged |
| Calculation questions ("how many times", "how long since") are not declined by the reader | `recall.answer_check.calculation_questions` | wrong declines 52 → 49 / 760, traps unchanged | conv-44 wrong declines 9 → 5 / 123 |
| Answers that are not statements about the asked person do not count: the reader's span lies only inside a question, is a speaker's name used as a greeting, or comes from the asked person's message that speaks only to the listener (you/your, no I/my) | `recall.answer_check.reject_non_statements` | traps declined 56 → 77 / 237; answerable declined 49 → 60 / 760 (9 had accepted a non-answer, 2 lost a right one) | dev, Gemini unchanged |
| A relative-date answer ("last weekend") is given as its resolved date ("the weekend before 17 July 2023") | always (with `memory.resolve_dates`) | time-question answers matching the gold 54 → 58 / 156 (`date_resolution/measure.py`) | retrieval unchanged |
| Top-level `answer`: the reader's most confident accepted span (a later, different answer is reported as `later_answer`) | `recall.answer_conflicts.best_answer` | answer token-F1 0.331 (first memory's span) → 0.375; time questions with the gold date 59 → 68 / 156 (`locomo_dev_answers.json`) | dev, Gemini identical |
| Deep recall: re-ranker (cross-encoder, 22M parameters) over the 50 candidates after the first 3, reading each memory also after the message before it. Off by default; per request `deep: true` | `recall.rerank` | see below | see below |

Files: `locomo_dev_*`, `conv-44_*`, `dev_*`, `gemini_batches_*` (before / after the first two).

## Held-out confirmation (run once)

LoCoMo test conversations conv-47, conv-48, conv-49, conv-50 (conv-44 excluded: it was inspected),
829 questions, nothing tuned on them. Files: `heldout_*.json`.

| Setup | served recall@10 | multi-hop | temporal | open | single-hop | wrong declines | traps declined | median recall |
|---|---|---|---|---|---|---|---|---|
| Morning (before today's changes) | 0.597 | 0.376 | 0.611 | 0.328 | 0.688 | 55/649 | 49/174 | 177 ms |
| Default now (second pass + calculation questions) | 0.658 | 0.373 | 0.649 | 0.383 | 0.778 | 54/649 | 49/174 | 198 ms |
| Deep recall (`deep: true`) | **0.711** | **0.485** | **0.687** | **0.417** | **0.821** | 54/649 | 49/174 | 500 ms |

## Re-ranker variants (LoCoMo dev, served evidence recall@10)

| Variant | all | multi-hop | temporal | open | single-hop | wrong declines | traps declined |
|---|---|---|---|---|---|---|---|
| A: second pass (no re-ranker) | 0.706 | 0.376 | 0.817 | 0.280 | 0.821 | 49/760 | 56/237 |
| C: re-ranker, keep 3 | 0.716 | 0.465 | 0.846 | 0.303 | 0.796 | 49 | 56 |
| E: second pass + re-ranker, keep 5 | 0.722 | 0.463 | 0.853 | 0.326 | 0.803 | 49 | 56 |
| F: re-ranker reading context, keep 3 | 0.737 | 0.465 | 0.853 | 0.292 | 0.834 | 49 | 56 |
| **G: second pass + F** | **0.748** | 0.458 | **0.865** | 0.286 | **0.852** | 49 | 56 |
| D: re-ranker *before* the answer check | 0.756 | 0.438 | 0.884 | 0.296 | 0.865 | **26** | **38** ✗ |

D was rejected: 18 more trap questions answered. Every other variant runs after recall has decided
to answer, so declines and traps cannot change.

Offline (saved rankings, `cross-encoder/ms-marco-*`, keep 3, context on):

* Candidates read: 20 → 0.733, 30 → 0.748, 50 → 0.775, 80 → 0.790, 100 → 0.792 (temporal and
  open-domain dip at 100).
* Model: MiniLM-L-12 0.779 vs L-6 0.775 at twice the cost; TinyBERT-L-2 0.736 at 1/7 of the cost.
* Blending the re-ranker score with the first ranking's relevance: no gain at any weight.
* 8-bit dynamic quantization (torch): 6× slower on Apple silicon CPU and changes the top 10.

Latency (conv-44, median recall, free machine, Apple M4): no re-ranker 140 ms; CPU 30 candidates
411-486 ms, 50 candidates 611-671 ms; Apple GPU 343 / 449 ms; TinyBERT-L-2 at 50 candidates 174 ms
but no gain in the engine (0.656 vs 0.667); L-6 without context, inputs cut to 128 tokens: 294 ms.
Recall without the re-ranker is the product's speed claim, so the re-ranker is opt-in per request.

How the filter was found: the reader's accepted answer was recorded for every LoCoMo dev question.
Of 181 trap questions it answered, 177 were answered from the *asked* person's own messages, mostly
from questions ("What inspired it?"), greetings ("Thanks, Melanie!") and reactions to the other
person's news.

Held-out check of the filter (second use of conv-47..50, which played no part in designing it;
`heldout_default_with_filter.json`): traps declined 49 → 63 of 174 (28% → 36%); answerable
questions declined 54 → 65 of 649, of which 7 never had the evidence retrieved (their answers
were almost certainly wrong) and at most 4 may have been right.

Automatic deep recall (`recall.rerank.auto`, deep only for questions matching a "spread-out"
pattern, 22% of LoCoMo dev questions): multi-hop 0.375 → 0.400 against 0.444 with deep recall
on every question; mean recall 207 → 279 ms (every question deep: 494 ms; Apple GPU). Left off.

## Tried and rejected

* **Serving each memory with the message before it, for the answering model** (LoCoMo conv-44, already
  development data; same gpt-4o-mini answer model and judge as the Mem0 comparison, Mem0's saved answers
  reused; `eval/mem0_comparison/run_recallix.py --context-out`): overall 67.7% either way, trick questions
  declined 31 -> 30 of 35, multi-hop +1 question. The trick questions Recallix still answers swap who owns or did
  something ("Audrey and Buddy" when Buddy is Andrew's dog); Mem0 stores self-contained facts ("Andrew's dog
  Buddy"), so its answering model notices. Next: ownership-aware facts (Phase 6 extractor; an ownership rule).

* **Reading the previous message only for short candidates** (`recall.rerank.context_max_words`, left at
  `null` = all), LoCoMo dev with deep recall on every question (`locomo_dev_deep_context_*.json`): served
  recall@10 0.739 (all) -> 0.718 (<= 15 words) / 0.717 (<= 10 words), single-hop 0.840 -> 0.796 / 0.793;
  median recall 326 -> 247 / 252 ms (Apple GPU). Longer messages gain from the previous message too.

* **Spreading results across sessions** (cap per session): recall@10 0.673 → 0.573-0.667.
* **Store-time neighbour context** (`recall.neighbour_context`): no gain on conv-44 once the
  second pass exists.
* **"Mention rule"** (another speaker's statement counts only if it names the asked person or
  says "you"): never fired. The trap questions are answered from the *asked* person's own
  unrelated messages, not from the other speaker's.
* **Re-ranker relevance as an answer gate** (an answer counts only if the accepted memory also
  scores ≥ −5 with the re-ranker): LoCoMo dev traps declined 56 → 94 for 11 real answers lost,
  but on the Gemini texting data first-answer accuracy fell 0.84 → 0.75 and wrong declines
  doubled (12% → 24%). The re-ranker (trained on web search queries) does not recognise that
  "started at apollo greams road" answers "Which hospital does she work at?".
* **Calculation-question pattern, wide version** (`how many|much|long|often ...`): let a trap
  through ("How many total employees work at the rescue shelter?"); narrowed to counting and
  time-between questions.

## Keeping the results on a decline (LongMemEval, evening)

`recall.keep_results_on_decline` (off by default): when recall declines, it still returns the
ranked results, marked `ABSTAINED` with `answer: null`. The decision is unchanged, so declines are
identical (56.7% of unanswerable, 16.6% of answerable), and so are our dev and test splits
(first-answer accuracy, answers, declines). `eval/run_eval.py` now reads a decline from `status`
rather than from an empty result list (identical numbers with the setting off).

LongMemEval_S, evidence recall@10 of what recall returns (files
`../longmemeval_s_bge-small-en-v1.5_keep-on-decline*.json`; baselines copied from the default run):

| Setup | overall | multi-session | temporal | time per question |
|---|---|---|---|---|
| Default | 0.727 | 0.559 | 0.622 | 0.56 s |
| Results kept on decline | 0.838 | 0.718 | 0.786 | 0.44 s |
| Results kept on decline + deep recall | **0.874** | **0.808** | **0.833** | 0.91 s |
| Dense vectors (same embeddings) | 0.798 | 0.737 | 0.674 | |

Rejected: also running the second pass and deep re-ranking on the kept results of a decline.
0.831 without deep recall (vs 0.838), 0.878 with it (vs 0.874, about 2 of 500 questions) at 18%
more time.

The setting stays off by default: an agent that ignores `status` would answer from low-confidence
results, and a wrong answer is worse than a refusal.
