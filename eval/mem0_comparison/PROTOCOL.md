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

## Amendment 1 (2026-11-24, before any run)

Corrections from reading mem0ai 2.2.1's source, plus one Recallix change measured on dev data only:

- **Dates for Mem0:** Mem0's open-source `add()` rejects `timestamp` ("Platform-only temporal parameter. Not
  supported in OSS"). Mem0 therefore gets each message's date only inside the text (`[session date] Speaker:
  message`), the same text Recallix gets.
- **Dates for Recallix:** Recallix also receives each message's session time as its `timestamp`, which its API
  accepts, plus one second per turn to keep order.
  - Recallix has resolved relative dates since 2026-11-24 (`memory.resolve_dates`, on by default).
  - Measured on LoCoMo **dev** only (eval/results/date_resolution/).
  - The answer passed to the answer model is each memory's `content_resolved` when present, else `content`:
    what Recallix returns to any caller.
- **Mem0 search:** `m.search(question, filters={"user_id": conv}, top_k=10)`, which is the 2.2.1 signature
  (`limit` is not a parameter). The default threshold of 0.1 is kept.

## Amendment 2: calibration and scope (2026-11-24, before the main run)

- **Calibration** (conv-44, first 40 messages, Mem0 storing): $0.057, 111 OpenAI calls, ~8,400 input tokens per
  message, median 2.2 s per `add()`, 0 errors. Mem0's prompts grow with the number of stored memories, so later
  messages cost more.
- **Scope** under the $2.60 projection rule: the first **2** test conversations (conv-44, conv-47), with 1,364 messages
  and 348 questions (273 answerable, 75 adversarial). The owner kept the $3 cap (not raised).
- **Mem0 install:** `mem0ai[nlp,extras]` (spaCy and fastembed: local, no API cost). Without them Mem0 warns that its
  BM25 keyword search and lemmatisation are disabled, which would handicap it.
- **The calibration's $0.057 counts toward the cap.** The main run stores into fresh storage.

## Amendment 3: aborted first start (2026-11-24)
- **What happened:** the first main-run start was stopped after about 2 minutes, with no results looked at. The
  Recallix side's offline setting (`HF_HUB_OFFLINE=1`) had leaked into the Mem0 processes, so Mem0 could not load its
  BM25 keyword-search model and would have run handicapped.
- **Cost:** about $0.07 was spent and counts toward the cap.
- **Fix:** the restart uses fresh storage, with the offline setting applied to the Recallix process only. Mem0 was
  verified to load BM25 first.

## Amendment 4: cost overrun on conv-47 (2026-11-24, during storage, before any answers)
- **Cost per message:** Mem0 cost more than the calibration showed (about $0.0019 and rising, against $0.0014),
  because its prompts grow with stored memories. Storing both conversations would have reached the $2.85 stop
  with nothing left for answering.
- **conv-47:** stopped after 80 of 689 messages. Its storage is saved and can be resumed.
- **Judging scope:** only conversations both systems fully completed are judged, so Recallix is never scored on
  questions Mem0 did not get.
