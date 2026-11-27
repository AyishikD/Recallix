# Recallix memory for Open WebUI

A filter that gives every chat in [Open WebUI](https://openwebui.com) long-term memory from your own Recallix
server. Each user's memories are kept apart, nothing leaves your machines, and no LLM is called to store or
recall anything.

For every message a user sends, the filter:

1. **recalls** what that user told any model before that bears on the new message,
2. gives the model those memories as a short system note, and shows them under the reply as **sources**;
   when Recallix has no answering memory, it tells the model to say so instead of guessing,
3. **stores** the new message, after recalling, so a question does not recall itself.

If Recallix is down or slow, the chat carries on without memory and shows a status line.

## Install (about 5 minutes)

1. **Run Recallix** ([quickstart](../../docs/quickstart.md)) and create one key with role `service`, which may
   act for every Open WebUI user:

   ```bash
   python -m backend.tools.keys create --user open-webui --role service
   ```

2. **Add the filter** in Open WebUI: *Admin Panel → Functions → Import from URL*, and paste

   ```
   https://github.com/AyishikD/Recallix/blob/main/integrations/open-webui/recallix_memory_filter.py
   ```

   (or *Create*, and paste the file's contents). Open WebUI installs its one requirement, `requests`.

3. **Set the valves** (the gear icon on the function):
   * `recallix_url`: where Recallix listens. If Open WebUI runs in Docker and Recallix on the same machine,
     use `http://host.docker.internal:8000`.
   * `api_key`: the service key from step 1.

4. **Turn it on**: toggle the function *Active*, then in its ⋮ menu choose *Global* for every model, or add it to
   specific models under *Workspace → Models → Filters*.

Check with two messages: "I moved to Pune in March for my new job.", then in a **new chat** "Where did I move?".
The reply should mention Pune, with the memory shown as a source.

## Settings

| Valve | Default | What it does |
|---|---|---|
| `mode` | `balanced` | How readily recall answers: `cautious` checks every answer twice (more "I don't know"), `helpful` answers more counting, ordering and "how long between" questions and adds the worked-out result. |
| `limit` | 5 | Memories given to the model per message. |
| `store_user_messages` | on | Store what users say. |
| `store_assistant_replies` | off | Also store the model's replies. Off by default: memories are what the user said, not what a model guessed. |
| `tell_model_on_decline` | on | When nothing answers, tell the model not to guess about the user's past. |
| `show_sources` | on | Show recalled memories as sources under the reply. |
| `agent_id` | `open-webui` | Recallix agent id; memories written here are read back here. |

Each user can switch memory off for themselves, or pick their own `mode`, in their user settings for the
function (*UserValves*).

## Good to know

* Users are told apart by their Open WebUI user id; a user's memories are only ever recalled for that user.
* Open WebUI does not run filters for its background tasks (chat titles, tags, follow-up suggestions), so those
  are never stored.
* The first message after Recallix starts can take up to a minute while its local models load; later ones take
  well under a second. Raise `timeout_seconds` if the first one times out.
* Tests: `tests/test_open_webui_filter.py` runs the filter against a real Recallix API.

Licence: Apache-2.0, like the SDKs.
