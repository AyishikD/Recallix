# recallic-adk: Recallic memory for Google ADK agents

*Formerly `recallix-adk` (0.1.x); renamed with the project, nothing else changed.*

`RecallicMemoryService` plugs a self-hosted [Recallic](https://github.com/Recallic/Recallic) server into
[Google's Agent Development Kit](https://google.github.io/adk-docs/) as the agent's memory service.

* **No LLM calls for memory.** Storing and searching use Recallic's local models; your agent's model is the
  only one you pay for.
* **Honest search.** When Recallic finds nothing that answers the query, `search_memory` returns no memories,
  so the agent is not handed near-misses to guess from.
* **Real dates.** Each event is stored with the time it happened, even when ADK hands over a session later.
* **Your data stays on your server**, separated per ADK user and per app.

## Install

```bash
pip install recallic-adk          # brings google-adk and the recallic SDK
```

Run Recallic ([quickstart](../../docs/quickstart.md)) and create one key with role `service`, which acts for
each of your ADK users:

```bash
python -m backend.tools.keys create --user my-adk-app --role service
```

## Use

```python
import os

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from recallic_adk import RecallicMemoryService

memory = RecallicMemoryService(api_key=os.environ["RECALLIC_SERVICE_KEY"],
                               base_url="http://localhost:8000",
                               mode="balanced")            # or "cautious" / "helpful"
agent = Agent(name="assistant", model="gemini-2.5-flash", instruction="...", tools=[load_memory])
runner = Runner(agent=agent, app_name="my_app", session_service=InMemorySessionService(), memory_service=memory)

# after a conversation (or after each turn):
await memory.add_session_to_memory(session)
```

| Argument | Default | What it does |
|---|---|---|
| `mode` | server's `recall.mode` | How readily search answers: `cautious` checks every answer twice; `helpful` answers more counting, ordering and "how long between" questions and adds a worked-out result as the first memory. |
| `limit` | 10 | Memories per search. |
| `store_authors` | `("user",)` | Which event authors are stored. The default stores what users said; `None` stores every author, including the agent's replies. |

What maps to what:

| ADK | Recallic |
|---|---|
| `user_id` | user (a service key acts for each user) |
| `app_name` | agent id, so each app reads back what it wrote |
| `add_session_to_memory` / `add_events_to_memory` / `add_memory` | `store`, one memory per text event, with its original time; storing a session again stores nothing twice |
| `search_memory` | `recall`; `MemoryEntry.custom_metadata` carries the relevance, trust and receipt |
| `delete_memory` | `delete`, with a deletion receipt in Recallic |

Needs Recallic server 0.2.5 or newer (for `mode` and original timestamps). A session is stored in one request per
100 events on server 0.2.6 or newer, and one request per event before that.

## Tests

Against a running Recallic: `RECALLIC_URL=http://127.0.0.1:8000 RECALLIC_SERVICE_KEY=rx_... pytest tests`.

Licence: Apache-2.0.
