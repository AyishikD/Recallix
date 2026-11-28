/* The integration pages (/integrations/<slug>). Each entry mirrors integrations/<dir>/README.md (or, for MCP,
   docs/quickstart.md section 8): keep them in step when an integration changes. */

export type Integration = {
    slug: string;
    name: string;            // the framework or app, as people search for it
    title: string;           // page <h1> and <title>
    description: string;     // meta description: what someone searching would want to know
    lede: string;
    packageName: string;     // what to install
    install: string;
    codeLabel: string;
    code: string;
    points: string[];
    settings: [string, string, string][];   // option, default, what it does
    readme: string;          // path in the repo
    registry?: { label: string; href: string };
};

const REPO = "https://github.com/Recallic/Recallic/blob/main";

export const INTEGRATIONS: Integration[] = [
    {
        slug: "open-webui",
        name: "Open WebUI",
        title: "Long-term memory for Open WebUI",
        description: "A filter that gives every Open WebUI chat long-term memory from your own Recallic server: it recalls what each user said before, says when there is no record, and needs no LLM calls for memory.",
        lede: "One filter, no code. Every chat remembers what each user told any model before, and when nothing answers, the model is told there is no record instead of guessing.",
        packageName: "recallic_memory_filter.py",
        install: "Admin Panel → Functions → Import from URL",
        codeLabel: "Setup",
        code: `# 1. On the Recallic server: one key that acts for every Open WebUI user
python -m backend.tools.keys create --user open-webui --role service

# 2. Open WebUI → Admin Panel → Functions → Import from URL:
https://github.com/Recallic/Recallic/blob/main/integrations/open-webui/recallic_memory_filter.py

# 3. Valves (gear icon): recallic_url, api_key  →  toggle Active, then Global`,
        points: [
            "Recalls what this user said in earlier chats, with any model, and gives it to the model as a short note; the memories appear as sources under the reply.",
            "When nothing answers, the model is told there is no record, so it does not invent the user's past.",
            "Stores each new message after recalling, so a question never recalls itself.",
            "Each user's memories stay apart; if Recallic is down, the chat carries on without memory.",
        ],
        settings: [
            ["mode", "balanced", "cautious checks every answer twice; helpful answers more counting, ordering and time questions"],
            ["limit", "5", "memories given to the model per turn"],
            ["store_assistant_replies", "off", "also store the model's replies (off: memories are what the user said)"],
            ["tell_model_on_decline", "on", "when nothing answers, tell the model not to guess"],
        ],
        readme: `${REPO}/integrations/open-webui/README.md`,
    },
    {
        slug: "llama-index",
        name: "LlamaIndex",
        title: "Long-term memory for LlamaIndex agents",
        description: "RecallicMemoryBlock plugs a self-hosted Recallic server into LlamaIndex's Memory: agents recall what each user said in earlier sessions and decline honestly when nothing answers. No LLM calls for memory.",
        lede: "A memory block for LlamaIndex's Memory. Each turn it recalls what this user said in any earlier session, and stores new messages right away.",
        packageName: "llama-index-memory-recallic",
        install: "pip install llama-index-memory-recallic",
        codeLabel: "Python",
        code: `import os
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.memory.recallic import RecallicMemoryBlock

block = RecallicMemoryBlock(api_key=os.environ["RECALLIC_SERVICE_KEY"],
                            base_url="http://localhost:8000", user_id="alice")
memory = Memory.from_defaults(session_id="alice-chat-1", memory_blocks=[block])

agent = FunctionAgent(llm=llm, tools=[...])
await agent.run("I moved to Pune in March.", memory=memory)

# days later, a new session with the same user
later = Memory.from_defaults(session_id="alice-chat-2", memory_blocks=[block])
await agent.run("Which city did I move to?", memory=later)   # "You moved to Pune."`,
        points: [
            "Recalls on the user's latest message, across sessions and agents, and adds it to the agent's context.",
            "When nothing answers, the agent is told there is no record instead of being handed near-misses.",
            "Stores new user messages every turn, not only when they fall out of short-term memory.",
            "Tested with a real FunctionAgent: a new session answered from memory and declined a question never asked.",
        ],
        settings: [
            ["user_id", "none", "whose memory this is (needed with a service key)"],
            ["mode", "server's", "cautious, balanced or helpful"],
            ["store_immediately", "True", "store each turn; False stores only what LlamaIndex moves out of short-term memory"],
            ["tell_model_on_decline", "True", "when nothing answers, tell the model not to guess"],
        ],
        readme: `${REPO}/integrations/llama-index/README.md`,
        registry: { label: "PyPI", href: "https://pypi.org/project/llama-index-memory-recallic/" },
    },
    {
        slug: "vercel-ai-sdk",
        name: "Vercel AI SDK",
        title: "Memory for the Vercel AI SDK",
        description: "recallicMemory() is AI SDK middleware that gives any model a self-hosted memory: it recalls what each user said before, declines honestly when nothing answers, and works with generateText, streamText and tools.",
        lede: "Wrap any model once. Before each call it recalls what this user said before; one wrapped model serves every user of your app.",
        packageName: "recallic-ai-sdk",
        install: "npm install recallic-ai-sdk ai",
        codeLabel: "TypeScript",
        code: `import { generateText, wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { recallicMemory } from "recallic-ai-sdk";

const model = wrapLanguageModel({
    model: openai("gpt-4.1-mini"),
    middleware: recallicMemory({ apiKey: process.env.RECALLIC_SERVICE_KEY!, baseUrl: "http://localhost:8000" }),
});

const { text } = await generateText({
    model,
    prompt: "Which city did I move to?",
    providerOptions: { recallic: { userId: "alice" } },   // whose memory, per call
});`,
        points: [
            "Memories go in as a system message after your own system prompt; when nothing answers, the model is told there is no record.",
            "Works with generateText, streamText, tools and multi-step calls: a tool loop recalls once and stores once.",
            "If Recallic is unreachable, the call goes on without memory instead of failing.",
            "AI SDK 7 by default, AI SDK 6 with specificationVersion: \"v3\".",
        ],
        settings: [
            ["userId", "per call", "whose memory, fixed here or with providerOptions.recallic.userId"],
            ["mode", "server's", "cautious, balanced or helpful"],
            ["limit", "5", "memories given to the model"],
            ["tellModelOnDecline", "true", "when nothing answers, tell the model not to guess"],
        ],
        readme: `${REPO}/integrations/vercel-ai/README.md`,
        registry: { label: "npm", href: "https://www.npmjs.com/package/recallic-ai-sdk" },
    },
    {
        slug: "haystack",
        name: "Haystack",
        title: "A memory store for Haystack agents",
        description: "RecallicMemoryStore follows Haystack's MemoryStore protocol: agents remember what each user said, re-adding a conversation is safe, and searches return nothing rather than near-misses when nothing answers.",
        lede: "A Haystack MemoryStore backed by your own Recallic server. The agent searches before it answers and adds the conversation after.",
        packageName: "recallic-haystack",
        install: "pip install \"recallic-haystack[agent]\"",
        codeLabel: "Python",
        code: `from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.agents import Agent
from recallic_haystack import RecallicMemoryStore

store = RecallicMemoryStore(base_url="http://localhost:8000")   # key from RECALLIC_API_KEY
agent = Agent(chat_generator=OpenAIChatGenerator(), memory_store=store,
              system_prompt="You are a helpful assistant.")

agent.run(messages=[ChatMessage.from_user("I moved to Pune in March.")],
          memory_store_kwargs={"user_id": "alice"})
agent.run(messages=[ChatMessage.from_user("Where did I move?")],
          memory_store_kwargs={"user_id": "alice"})`,
        points: [
            "Follows the MemoryStore protocol of haystack-experimental: add, search, delete, and serialization with a Secret.",
            "When nothing answers, search returns no memories instead of near-misses for the model to guess from.",
            "The agent hands over the whole conversation after each run; Recallic stores each message once.",
            "Tested with a real Haystack Agent run (haystack-ai 2.23 with haystack-experimental 0.19).",
        ],
        settings: [
            ["agent_id", "haystack", "Recallic agent id; each agent reads back what it wrote"],
            ["mode", "server's", "cautious, balanced or helpful"],
            ["store_roles", "(\"user\",)", "which message roles are stored"],
        ],
        readme: `${REPO}/integrations/haystack/README.md`,
        registry: { label: "PyPI", href: "https://pypi.org/project/recallic-haystack/" },
    },
    {
        slug: "google-adk",
        name: "Google ADK",
        title: "A memory service for Google ADK agents",
        description: "RecallicMemoryService is a self-hosted memory service for Google's Agent Development Kit: sessions are stored with their real dates and searches return nothing when nothing answers. No LLM calls for memory.",
        lede: "Plug Recallic into an ADK Runner as its memory service. Sessions go in with their real dates; load_memory gets back only what answers.",
        packageName: "recallic-adk",
        install: "pip install recallic-adk",
        codeLabel: "Python",
        code: `import os
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from recallic_adk import RecallicMemoryService

memory = RecallicMemoryService(api_key=os.environ["RECALLIC_SERVICE_KEY"],
                               base_url="http://localhost:8000")
agent = Agent(name="assistant", model="gemini-2.5-flash", instruction="...", tools=[load_memory])
runner = Runner(agent=agent, app_name="my_app", session_service=InMemorySessionService(),
                memory_service=memory)

# after a conversation:
await memory.add_session_to_memory(session)`,
        points: [
            "Maps ADK's user_id to the Recallic user and app_name to the agent, so each app reads back what it wrote.",
            "Each event is stored with the time it happened, even when ADK hands the session over later.",
            "When nothing answers, search_memory returns no memories, so the agent is not handed near-misses.",
            "Storing a session again stores nothing twice; sessions go in one request per 100 events.",
        ],
        settings: [
            ["mode", "server's", "cautious, balanced or helpful"],
            ["limit", "10", "memories per search"],
            ["store_authors", "(\"user\",)", "which event authors are stored; None stores every author"],
        ],
        readme: `${REPO}/integrations/google-adk/README.md`,
        registry: { label: "PyPI", href: "https://pypi.org/project/recallic-adk/" },
    },
    {
        slug: "mcp",
        name: "MCP (Claude Desktop, Cursor)",
        title: "An MCP memory server for Claude Desktop and Cursor",
        description: "recallic-mcp gives Claude Desktop, Cursor or any MCP client one long-term memory that you own: recall, store, correct and forget, shared across clients, with honest declines.",
        lede: "Give Claude Desktop, Cursor or any MCP client the same memory. What you tell one client, the others can recall.",
        packageName: "recallic",
        install: "pip install recallic",
        codeLabel: "MCP client config (mcpServers)",
        code: `"mcpServers": {
  "recallic": {
    "command": "recallic-mcp",
    "env": { "RECALLIC_API_KEY": "rx_...", "RECALLIC_AGENT_ID": "claude-desktop" }
  }
}`,
        points: [
            "Tools: recall_memory, store_memory, update_memory, forget_memory, relevant_memories, confirm_answer, export_memories.",
            "When a recall finds nothing, the client is told to say it doesn't know rather than guess.",
            "Corrections update the memory instead of piling up contradicting versions.",
            "Memories stored as shared are visible to every client of the same user.",
        ],
        settings: [
            ["RECALLIC_API_KEY", "required", "your Recallic key"],
            ["RECALLIC_URL", "http://localhost:8000", "where Recallic runs"],
            ["RECALLIC_AGENT_ID", "mcp", "name for memories written through this client"],
        ],
        readme: `${REPO}/docs/quickstart.md`,
        registry: { label: "PyPI", href: "https://pypi.org/project/recallic/" },
    },
];

export function integration(slug: string) {
    return INTEGRATIONS.find((i) => i.slug === slug);
}
