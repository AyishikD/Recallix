// node --test test/memory.test.mjs  (after npm run build). Live tests need RECALLIX_URL and RECALLIX_SERVICE_KEY.
import assert from "node:assert/strict";
import { test } from "node:test";
import { generateText, wrapLanguageModel, tool, stepCountIs } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { z } from "zod";
import mod from "../dist/index.js";
const { recallixMemory, MARKER } = mod;

const result = (text) => ({
    content: [{ type: "text", text }], finishReason: { unified: "stop", raw: undefined },
    usage: { inputTokens: { total: 1, noCache: 1 }, outputTokens: { total: 1, text: 1 } }, warnings: [],
});

/** A model that records each prompt it is given. */
function recordingModel(replies = ["ok"]) {
    const prompts = [];
    let i = 0;
    const model = new MockLanguageModelV4({
        doGenerate: async ({ prompt }) => { prompts.push(prompt); return replies[Math.min(i++, replies.length - 1)]; },
    });
    return { model, prompts };
}

function fakeClient(recallOut) {
    const calls = [];
    return {
        calls,
        recall: async (q, o) => { calls.push(["recall", q, o]); if (recallOut instanceof Error) throw recallOut; return recallOut; },
        store: async (c, o) => { calls.push(["store", c, o]); return { status: "stored" }; },
    };
}

const FOUND = { status: "CONSISTENT", memories: [{ id: 1, content: "I moved to Pune in March.", timestamp: 1709251200 }] };
const systemTexts = (prompt) => prompt.filter((m) => m.role === "system").map((m) => m.content);

test("memories go in as a system message after the app's own, then the message is stored", async () => {
    const client = fakeClient(FOUND);
    const { model, prompts } = recordingModel([result("You moved to Pune.")]);
    const wrapped = wrapLanguageModel({ model, middleware: recallixMemory({ apiKey: "x", client }) });
    await generateText({ model: wrapped, system: "You are helpful.", prompt: "Where did I move?",
                         providerOptions: { recallix: { userId: "alice" } } });
    const sys = systemTexts(prompts[0]);
    assert.equal(sys[0], "You are helpful.");
    assert.ok(sys[1].startsWith(MARKER) && sys[1].includes("[01 Mar 2024] I moved to Pune in March."));
    assert.deepEqual(client.calls.map((c) => c[0]), ["recall", "store"]);  // recall first: the question does not recall itself
    assert.equal(client.calls[0][2].userId, "alice");
    assert.equal(client.calls[1][1], "Where did I move?");
});

test("no record: the model is told not to guess; can be turned off", async () => {
    for (const tell of [true, false]) {
        const { model, prompts } = recordingModel([result("ok")]);
        const wrapped = wrapLanguageModel({ model, middleware: recallixMemory({ apiKey: "x", tellModelOnDecline: tell,
                                                                                client: fakeClient({ status: "ABSTAINED", memories: [] }) }) });
        await generateText({ model: wrapped, prompt: "What is my brother's name?" });
        const sys = systemTexts(prompts[0]).join("\n");
        assert.equal(sys.includes("rather than guess"), tell);
    }
});

test("a tool loop recalls once and stores once", async () => {
    const client = fakeClient(FOUND);
    let step = 0;
    const prompts = [];
    const model = new MockLanguageModelV4({
        doGenerate: async ({ prompt }) => {
            prompts.push(prompt);
            if (step++ === 0) return { ...result(""), content: [{ type: "tool-call", toolCallId: "t1", toolName: "today", input: "{}" }],
                                       finishReason: { unified: "tool-calls", raw: undefined } };
            return result("It is the 27th.");
        },
    });
    const wrapped = wrapLanguageModel({ model, middleware: recallixMemory({ apiKey: "x", client }) });
    await generateText({ model: wrapped, prompt: "What's the date?", stopWhen: stepCountIs(3),
                         tools: { today: tool({ inputSchema: z.object({}), execute: async () => "2026-11-27" }) } });
    assert.equal(prompts.length, 2);
    assert.ok(systemTexts(prompts[1]).some((s) => s.startsWith(MARKER)));  // the memory is in every step
    assert.deepEqual(client.calls.map((c) => c[0]), ["recall", "store"]);
});

test("Recallix down: the call goes on without memory and nothing is stored", async () => {
    const errors = [];
    const client = fakeClient(new Error("ECONNREFUSED"));
    const { model, prompts } = recordingModel([result("ok")]);
    const wrapped = wrapLanguageModel({ model, middleware: recallixMemory({ apiKey: "x", client, onError: (e) => errors.push(e) }) });
    const out = await generateText({ model: wrapped, prompt: "hi" });
    assert.equal(out.text, "ok");
    assert.equal(systemTexts(prompts[0]).length, 0);
    assert.equal(errors.length, 1);
    assert.deepEqual(client.calls.map((c) => c[0]), ["recall"]);
});

test("disabled per call", async () => {
    const client = fakeClient(FOUND);
    const { model } = recordingModel([result("ok")]);
    const wrapped = wrapLanguageModel({ model, middleware: recallixMemory({ apiKey: "x", client }) });
    await generateText({ model: wrapped, prompt: "hi", providerOptions: { recallix: { disabled: true } } });
    assert.equal(client.calls.length, 0);
});

const URL = process.env.RECALLIX_URL, KEY = process.env.RECALLIX_SERVICE_KEY;
test("live: told in one conversation, recalled in the next; never told is declined", { skip: !(URL && KEY) }, async () => {
    const user = `vai-${Math.random().toString(36).slice(2, 10)}`;
    const { model, prompts } = recordingModel([result("Noted.")]);
    const wrapped = wrapLanguageModel({ model, middleware: recallixMemory({ apiKey: KEY, baseUrl: URL }) });
    const opts = { providerOptions: { recallix: { userId: user } } };
    await generateText({ model: wrapped, prompt: "My cat is called Biscuit and I moved to Pune in March.", ...opts });
    await generateText({ model: wrapped, prompt: "What is my cat called?", ...opts });
    await generateText({ model: wrapped, prompt: "What is my brother's name?", ...opts });
    assert.ok(systemTexts(prompts[1]).join("\n").includes("Biscuit"));
    const third = systemTexts(prompts[2]).join("\n");
    assert.ok(third.includes("rather than guess") && !third.includes("Biscuit"));
});
