# Recallix vs Mem0 on LoCoMo: protocol

Written 2026-11-23, before any run. Not edited after results exist (changes would be appended with a reason).

## Question
On the same conversations and questions, how do Recallix and Mem0 compare on:

- answer accuracy;
- declining questions that have no answer;
- cost to store;
- speed?

## Data
- **Conversations:** LoCoMo (locomo10.json), the held-out **test** split, conversations 5-9 (conv-44, -47, -48, -49, -50):
  3,122 messages and 987 questions. Of the questions, 763 are answerable (categories 1-4) and 224 are adversarial
  (category 5, no answer in the conversation).
- **Who has seen it:** Recallix was never tuned on this split. Mem0 is used as shipped.
- **Input text:** every message is given to both systems as the same text, `[session date] Speaker: message`
  (plus any image caption), as eval/benchmarks/locomo.py builds it. Both therefore see the dates in the text.
  - This avoids a known pitfall: Mem0 otherwise stamps memories with today's date (mem0 issue #3944).

## Systems
- **Recallix:** as shipped (config.yaml, 2026-11-23): local embeddings BAAI/bge-small-en-v1.5, answer check on,
  no LLM. One user per conversation. Recall `limit=10`.
- **Mem0:** open source `mem0ai` 2.2.1, `Memory.from_config` with library defaults except:
  - LLM `gpt-4o-mini`. The library default is `gpt-5-mini`. gpt-4o-mini is what Mem0's own benchmark
    harness (github.com/mem0ai/memory-benchmarks) runs its open-source server with, and it fits the budget.
  - Embedder `text-embedding-3-small` (the library default).
  - Local Qdrant and history database in a scratch folder.
  - Telemetry off.
  - Each message is added as one user message (`m.add([{"role": "user", "content": text}], user_id=conv)`),
    with inference left on (the default). Search uses `limit=10`.

## Answering and judging (identical for both systems)
- **Answer model:** `gpt-4o-mini`, temperature 0, with this prompt:
  - system: "You answer questions about a user's conversations using only the memories provided. If the memories
    do not contain the answer, reply exactly: I don't know. Otherwise answer in a short phrase."
  - user: "Memories:\n- <memory 1>\n- ...\n\nQuestion: <question>\nAnswer:"
  - A system that returns no memories (Recallix ABSTAINED / NO_MATCH) gets the answer "I don't know" with no call.
- **Answerable questions (categories 1-4):**
  - Judge `gpt-4o-mini`, temperature 0: "Question, gold answer, generated answer. Is the generated answer
    correct, i.e. it states the same fact as the gold answer (dates and numbers may be written differently)?
    Reply CORRECT or WRONG."
  - "I don't know" counts as WRONG with no call.
- **Adversarial questions (category 5):** correct means the answer is "I don't know" (case-insensitive prefix
  "i don't know" or "i do not know"). No judge call.

## Measured
- **Accuracy:** overall, per category, answerable only, and adversarial declined.
- **Cost to store:** LLM calls, tokens and dollars. Recallix: 0 LLM calls.
- **Speed:** median and p95 milliseconds to store one message and to recall one question. Wall-clock includes
  network for Mem0; Recallix runs on an Apple M4.
- **Answer and judge cost:** reported separately; this is the same for both systems by design.

## Budget
- **Hard cap:** $3 of OpenAI spend, enforced in code. Every OpenAI call (Mem0's internal ones included) is counted
  from the API's own token usage at published prices plus 10%. All calls stop at an estimated $2.85.
- **Scope:** fixed before the main run by a calibration on the first 40 messages of conv-44 (the cost per message
  for Mem0).
  - The run covers the largest number of test conversations, taken in the order above, whose projected cost
    (storage plus answering plus judging, for both systems) is at most $2.60.
  - Both systems are evaluated on exactly that scope.
  - If the cap still trips, results so far are reported as partial.

## Integrity
- Each system is run once.
- No prompt, threshold or setting is changed after seeing results.
- Results are published whichever way they fall, including categories where Mem0 is better.
