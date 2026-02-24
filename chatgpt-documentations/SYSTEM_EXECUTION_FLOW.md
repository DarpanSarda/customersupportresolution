🧠 AI Agent Framework — Execution & Sequencing Flow
🎯 Purpose

This document defines the correct architectural sequencing and execution flow for the reusable agent framework.

It ensures:

No premature complexity

No over-engineering

Stable layering

Clean progression toward enterprise readiness

🏗 System Architecture Overview
FastAPI (Chat Route)
        ↓
GraphEngine (Session-aware DAG)
        ↓
Orchestrator (Deterministic Controller)
        ↓
Agents (Intent → Decision → Response)
        ↓
Patch → Validator → StateManager
        ↓
Versioned State
📌 Phase 1 — Core Execution Kernel (Completed)
Implemented Components

✅ Patch Model (Immutable)

✅ PatchValidator

✅ StateManager (Snapshot + Versioning)

✅ Orchestrator (Single-Agent Execution)

✅ GraphValidator (DAG Enforcement)

✅ GraphEngine (Session-Aware)

✅ IntentAgent (LLM-Based)

✅ FastAPI /chat endpoint

✅ Bootstrap (Composition Root)

Current Flow
Chat → IntentAgent → Stop

At this stage, system only performs intent classification.

🟡 Phase 2 — Complete Core Agent Chain (Current Focus)
🎯 Goal

Build the full conversational agent pipeline before adding observability or advanced tooling.

Required Agents
1️⃣ IntentAgent

Reads conversation.latest_message

Classifies intent

Writes to understanding

2️⃣ DecisionAgent (Next to Build)

Reads understanding

Applies confidence threshold

Determines next action

Writes to decision

Example decision output:

{
  "action": "GENERATE_RESPONSE",
  "route": "STANDARD"
}
3️⃣ ResponseAgent

Reads decision

Generates final response (LLM-based)

Writes to execution.final_response

🔄 Target Execution Flow
Chat
  ↓
IntentAgent
  ↓
DecisionAgent
  ↓
ResponseAgent
  ↓
Return execution.final_response
🟡 Phase 3 — Introduce Tool Execution

After agent chain is stable:

Add Tool abstraction

Add one dummy tool

Write tool calls into execution.tools_called

Handle tool failures

Integrate tool retry logic

New flow example:

Intent
  ↓
Decision
  ↓
ToolExecution
  ↓
Response
🟢 Phase 4 — Integrate Observability (Correct Timing)
Why Not Earlier?

Observability should be added when:

Multiple agent hops exist

Tool calls exist

Failures propagate

Retry paths exist

Async jobs exist

Right now, the system is too simple.

Observability Integration Phase

Integrate:

OpenTelemetry spans

Langfuse tracing

LLM call instrumentation

Agent-level spans

Tool execution spans

Graph transition spans

This ensures:

Full trace visibility

Performance analysis

Debugging across hops

Replay-safe tracing

🚫 What We Are NOT Doing Yet

No async job handling

No dynamic graph mutation

No loop-based graph transitions

No distributed state

No multi-tenant config resolution

No advanced feature flag engine

No YAML-based config overrides

Keep system minimal until core behavior stabilizes.

🧭 Final Development Roadmap
Step 1

Build DecisionAgent (rule-based first)

Step 2

Build ResponseAgent (LLM-based)

Step 3

Wire full Intent → Decision → Response flow

Step 4

Update chat endpoint to return execution.final_response

Step 5

Add tool execution layer

Step 6

Integrate OpenTelemetry + Langfuse