# llama-index-memory-recallix: Recallix memory for LlamaIndex agents

`RecallixMemoryBlock` plugs a self-hosted [Recallix](https://github.com/AyishikD/Recallix) server into
[LlamaIndex](https://www.llamaindex.ai)'s `Memory` as a long-term memory block.

* **No LLM calls for memory.** Storing and searching use Recallix's local models.
* **Across sessions and agents.** Each turn, the block recalls what this user said before, in any earlier session,
  that bears on their latest message.
* **Honest.** When nothing answers it, the model is told there is no record instead of being handed near-misses to
  guess from.
* **Stored right away.** New user messages are stored every turn, not only when LlamaIndex moves them out of
  short-term memory; re-sending one Recallix already holds is skipped.
* **Your data stays on your server**, separated per user and per agent.

## Install

```bash
pip install llama-index-memory-recallix
```

Run Recallix ([quickstart](../../docs/quickstart.md)) and create a key with role `service`, which acts for each of
your users (or use a user's own key and leave `user_id` out):

```bash
python -m backend.tools.keys create --user my-llamaindex-app --role service
```

## Use

```python
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.memory.recallix import RecallixMemoryBlock

block = RecallixMemoryBlock(api_key=os.environ["RECALLIX_SERVICE_KEY"], base_url="http://localhost:8000",
                            user_id="alice", mode="balanced")       # or "cautious" / "helpful"
memory = Memory.from_defaults(session_id="alice-chat-1", memory_blocks=[block])

agent = FunctionAgent(llm=llm, tools=[...])
await agent.run("I moved to Pune in March.", memory=memory)

# days later, a new session: a new Memory, the same user
later = Memory.from_defaults(session_id="alice-chat-2", memory_blocks=[block])
await agent.run("Which city did I move to?", memory=later)   # "You moved to Pune."
```

| Parameter | Default | What it does |
|---|---|---|
| `user_id` | `None` | Whose memory this is. Needed with a `service` key; leave out with a user's own key. |
| `agent_id` | `llamaindex` | Recallix agent id; each agent reads back what it wrote (and what any agent shared). |
| `mode` | server's `recall.mode` | `cautious` checks every answer twice; `helpful` answers more counting, ordering and "how long between" questions and adds a worked-out result. |
| `limit` | 5 | Memories given to the model per turn. |
| `store_roles` | `("user",)` | Which message roles are stored. The default stores what users said, not the model's replies. |
| `store_immediately` | `True` | Store new messages every turn (after recalling). `False`: only when LlamaIndex moves them out of short-term memory. |
| `tell_model_on_decline` | `True` | When nothing answers, tell the model not to guess about the user's past. |

Notes:

* The block is queried with the user's latest message. A message's original time can be given as
  `additional_kwargs={"timestamp": <unix seconds>}`; the LlamaIndex session id is kept in the memory's metadata.
* The API key is not included when the block is serialized.
* Needs Recallix server 0.2.5 or newer; messages go in one request per 100 on server 0.2.6 or newer.
* Tested with llama-index-core 0.14.

## Tests

Against a running Recallix: `RECALLIX_URL=http://127.0.0.1:8000 RECALLIX_SERVICE_KEY=rx_... pytest tests`.
They include a real LlamaIndex chat engine run with a scripted model.

Licence: Apache-2.0.
