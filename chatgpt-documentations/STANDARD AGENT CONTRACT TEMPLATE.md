# Agent Contract — <AGENT_NAME>

## 1. Agent Purpose
Clearly define the single responsibility of this agent.

This agent:
- MUST only handle its assigned responsibility
- MUST NOT call other agents directly
- MUST only read from shared state
- MUST only return structured state updates

---

## 2. Input State Schema

This agent receives the following state fields:

{
  "client_id": "string",
  "session_id": "string",
  "user_message": "string",
  "conversation_history": [],
  "intent": "optional",
  "sentiment": "optional",
  "rag_results": [],
  "policy_results": {},
  "context_bundle": {},
  "config": {}
}

Specify which fields this agent actually uses.

---

## 3. Output State Update

This agent must return ONLY the fields it modifies:

Example:

{
  "intent": "...",
  "confidence": 0.92
}

OR

{
  "rag_results": [...],
  "rag_confidence": 0.88
}

Never overwrite unrelated state fields.

---

## 4. Config Dependencies

List which config values this agent requires.

Example:
- intent_labels
- escalation_threshold
- tone_style
- client_vector_collection

---

## 5. Tools Used

List tools this agent is allowed to call.
If none, explicitly state: "No tools."

---

## 6. Failure Handling

Define:
- What happens if low confidence?
- What happens if tool fails?
- What state flags are set?

Example:

{
  "error_flag": true,
  "error_type": "low_confidence"
}

---

## 7. Constraints

- Must not hallucinate actions.
- Must not override policy decisions.
- Must return structured JSON.
- Must be deterministic given same input.

---

## 8. Test Cases

Provide 3–5 sample inputs and expected structured outputs.
