# Agent Integration Status

## Overview
All agents are properly integrated with the tool registry system. This document summarizes the current state of each agent and their tool requirements.

---

## Agents Summary

### 1. IntentAgent ✅ COMPLETE
**File**: `agents/IntentAgent.py`

**Purpose**: Classifies user messages into predefined intents (GREETING, FAQ_QUERY, REFUND_REQUEST, etc.)

**Tool Requirements**: None
- Uses LLM directly for classification
- No external tools needed

**Tool Registry**: Not used
```python
IntentAgent(llm_client=llm, system_prompt=prompt)
```

**State Updates**:
- `intent`: Detected intent label
- `intent_confidence`: Classification confidence
- `intent_raw`: Raw classification data

---

### 2. SentimentAgent ✅ COMPLETE
**File**: `agents/SentimentAgent.py`

**Purpose**: Analyzes user messages for emotional tone, urgency, and toxicity

**Tool Requirements**: None
- Uses LLM directly for sentiment analysis
- No external tools needed

**Tool Registry**: Not used
```python
SentimentAgent(llm_client=llm, system_prompt=prompt)
```

**State Updates**:
- `sentiment`: Detected sentiment label
- `sentiment_confidence`: Urgency score
- `sentiment_raw`: Raw sentiment data

---

### 3. RAGRetrievalAgent ✅ COMPLETE WITH FAQTool
**File**: `agents/RAGRetrievalAgent.py`

**Purpose**: Two-stage retrieval (FAQ lookup → Knowledge base search)

**Tool Requirements**: FAQTool
```python
RAGRetrievalAgent(llm_client=llm, tool_registry=tool_registry)
```

**Tool Permissions**: `["FAQTool"]`
- Can use FAQTool for FAQ lookups
- Two-stage retrieval:
  1. FAQ lookup via FAQTool (high confidence threshold: 0.85)
  2. Knowledge base vector search (fallback)

**State Updates**:
- `rag_results`: Retrieved passages
- `rag_context`: Formatted context string
- `rag_source_type`: "faq" or "knowledge_base"
- `rag_confidence`: Retrieval confidence score

**Tool Usage**:
```python
async def _faq_exact_match(query, tenant_id):
    result = await self.use_tool("FAQTool", {
        "query": query,
        "tenant_id": tenant_id,
        "top_k": 1,
        "threshold": 0.85
    })
```

---

### 4. ResponseAgent ✅ COMPLETE
**File**: `agents/ResponseAgent.py`

**Purpose**: Generates final conversational response based on gathered context

**Tool Requirements**: None (but tool_registry is passed for future extensibility)

**Tool Registry**: Passed but not currently used
```python
ResponseAgent(llm_client=llm, tool_registry=tool_registry)
```

**Tool Permissions**: `["FAQTool"]` (configured but not used)
- Has permission for FAQTool but doesn't call it directly
- Reads FAQ results from state (set by RAGRetrievalAgent)

**State Updates**:
- `response`: Generated response text
- `response_type`: "faq_answer", "kb_answer", "generic", or "escalation"

**Design Rationale**:
- RAGRetrievalAgent handles FAQTool calls
- Results stored in `rag_context` state
- ResponseAgent reads from state to generate response
- This maintains clear separation of concerns

---

### 5. PolicyAgent ✅ COMPLETE
**File**: `agents/PolicyAgent.py`

**Purpose**: Evaluate business rules and recommend actions

**Tool Requirements**: None
- Uses ConfigService to load policies from database
- Evaluates intent + sentiment against policy rules
- Returns business action recommendations

**Tool Registry**: Not used

```python
PolicyAgent(llm_client=llm, system_prompt=prompt, config_service=config_service)
```

**State Updates**:
- `policy_results`: Full policy evaluation results
- `policy_action`: Recommended business action
- `policy_raw`: Raw evaluation data for debugging

**Features**:
- Policy loading from database via ConfigService
- Intent matching to policy rules
- Sentiment threshold evaluation for escalation
- Required field validation
- Blocked sentiment detection
- Priority calculation
- Policy caching for performance

**Database**: Uses `policies` table with schema:
- `tenant_id`, `intent_type`, `business_action`
- `required_fields` (JSON), `escalate_if_sentiment_above`
- `blocked_sentiments` (JSON), `priority`

---

## Tool Registry Configuration

### Registered Tools

| Tool Name | Tool Class | Purpose |
|-----------|------------|---------|
| FAQTool | tools.FAQTool.FAQTool | FAQ lookup from Qdrant vector store |
| ApiTool | tools.ApiTool.ApiTool | Generic HTTP API caller with retry logic |

### Agent Permissions

| Agent | Allowed Tools | Usage |
|-------|---------------|-------|
| RAGRetrievalAgent | FAQTool | ✅ Active - Uses FAQTool for FAQ lookups |
| ResponseAgent | FAQTool | ⚸ Configured but not used (reads from state) |

### Permission Setup

```python
# In ChatService._create_tool_registry()
registry.set_agent_permissions("RAGRetrievalAgent", ["FAQTool"])
registry.set_agent_permissions("ResponseAgent", ["FAQTool"])
```

---

## Tool Execution Flow

### RAGRetrievalAgent FAQ Lookup

```
User Query
    ↓
RAGRetrievalAgent.process()
    ↓
_faq_exact_match(query, tenant_id)
    ↓
self.can_use_tool("FAQTool") → Permission check
    ↓
self.use_tool("FAQTool", payload)
    ↓
ToolRegistry.execute_tool(agent_name="RAGRetrievalAgent", tool_name="FAQTool", payload)
    ↓
FAQTool.execute(payload)
    ↓
Qdrant vector search in faq_{tenant_id} collection
    ↓
ToolResult.success(data={results, total_found, query, tenant_id})
    ↓
RAGRetrievalAgent processes result
    ↓
Returns FAQ answer OR falls through to knowledge base search
```

---

## Collection Naming

### Qdrant Collections

| Collection | Purpose | Managed By |
|------------|---------|------------|
| `faq_{tenant_id}` | FAQ embeddings | FAQTool |
| `knowledge_base_{tenant_id}` | Knowledge base embeddings | RAGRetrievalAgent |

---

## Debug Logging

All components have comprehensive debug logging:

### ChatService
- Tool creation and registration
- Agent permission setup
- Tool registry initialization

### RAGRetrievalAgent
- FAQ lookup attempts
- Tool availability checks
- FAQTool execution results

### Orchestrator
- Agent initialization
- Tool registry passing
- Agent execution order

---

## Next Steps

1. ✅ FAQTool integrated with RAGRetrievalAgent
2. ✅ ApiTool created and registered
3. ✅ Tool permission system working
4. ⏳ Test FAQTool with actual FAQ data in Qdrant
5. ⏳ Create FAQ ingestion endpoint
6. ⏳ Test complete end-to-end flow
7. ⏳ Remove debug logging (after testing complete)

---

## File Structure

```
customersupportresolution/
├── agents/
│   ├── BaseAgent.py              # Base class for all agents
│   ├── IntentAgent.py            # Intent classification
│   ├── SentimentAgent.py         # Sentiment analysis
│   ├── RAGRetrievalAgent.py      # RAG retrieval (uses FAQTool)
│   ├── PolicyAgent.py            # Policy evaluation (uses ConfigService)
│   └── ResponseAgent.py          # Response generation
├── tools/
│   ├── BaseTools.py              # Base class for all tools
│   ├── FAQTool.py                # FAQ lookup tool
│   ├── ApiTool.py                # HTTP API caller tool
│   └── __init__.py
├── core/
│   ├── ToolRegistry.py           # Tool and permission management
│   ├── Orchestrator.py           # Agent coordination
│   └── ConversationState.py      # Shared state management
├── database/
│   └── migrations/
│       └── create_policies_table.sql  # PolicyAgent database schema
├── prompts/
│   ├── PolicyAgent/
│   │   └── v1.txt                # PolicyAgent prompt template
│   └── ...
└── services/
    └── ChatService.py            # Main service with tool registry creation
```

---

## Verification Checklist

- [x] IntentAgent - Working, no tools needed
- [x] SentimentAgent - Working, no tools needed
- [x] RAGRetrievalAgent - Working with FAQTool integration
- [x] PolicyAgent - Working with ConfigService integration
- [x] ResponseAgent - Working, reads from state
- [x] ToolRegistry - Manages tools and permissions
- [x] FAQTool - Registered and functional
- [x] ApiTool - Registered and functional
- [x] Agent permissions - Configured correctly
- [ ] Policies table created in database
- [ ] Default policies inserted
- [ ] End-to-end test with FAQ data
- [ ] End-to-end test without FAQ data (KB fallback)
- [ ] End-to-end test with PolicyAgent
- [ ] FAQ ingestion endpoint
