When you finish building agents individually, use this checklist before integration.

🔲 1. Interface Stability

 Every agent has fixed input schema

 Every agent has fixed output fields

 No field name inconsistencies

 Confidence scores standardized (0–1 scale)

🔲 2. Responsibility Isolation

 Intent agent does not retrieve KB

 RAG agent does not generate final answer

 Policy agent does not speak conversationally

 Resolution agent does not change policy results

 Escalation agent does not create tickets directly

🔲 3. Config Validation

 All required config keys validated at load

 Missing config fails early

 Client isolation enforced

 Vector collection names correct per client

🔲 4. Tool Safety

 Tools never called directly by LLM

 Tool failures set error flags

 Tool timeouts handled

 Ticket system isolated per client

🔲 5. Escalation Logic

 Escalation thresholds configurable

 Angry sentiment triggers tested

 Low confidence triggers tested

 Retry limit enforced

🔲 6. Multi-Client Safety

 No cross-client vector search

 No cross-client memory leakage

 No shared ticket namespace

🔲 7. Observability

 Log per-agent input

 Log per-agent output

 Log tool calls

 Log latency