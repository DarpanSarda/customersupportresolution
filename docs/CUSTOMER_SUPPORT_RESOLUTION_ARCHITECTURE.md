# Customer Support Resolution System - Complete Architecture

## Executive Summary

This document defines the complete AI-powered Customer Support Resolution System architecture, including all agents, tools, permissions, and data flow. The system is designed as a **multi-tenant, reusable agent framework** that can be deployed across multiple clients without modifying core agent code.

---

## System Overview

### Architecture Principles
- **Agent-by-agent modular architecture** - Single responsibility per agent
- **Config-driven behavior** - No hardcoded business rules
- **Tool-based actions** - No hallucinated actions, only tool calls
- **Multi-client isolation** - Complete tenant separation
- **Vertical slice development** - Integrated from early stage
- **Prompt-driven logic** - Behavior controlled via prompts

### Technology Stack
- **LLM**: OpenAI, Groq, OpenRouter (via OpenAI-compatible)
- **Orchestration**: LangGraph-style graph engine
- **Vector DB**: For RAG knowledge retrieval
- **Database**: PostgreSQL/SQLite for configuration
- **Encryption**: AES-256-GCM for API keys

---

## Agent Architecture (8 Core Agents)

### Tier 1: Understanding Agents

| Agent | Purpose | Inputs | Outputs | Tools |
|-------|---------|--------|---------|-------|
| **IntentClassificationAgent** | Classify user intent | user_message, conversation_history | intent, confidence, tool_mapping | None (LLM) |
| **SentimentAgent** | Detect emotion & urgency | latest_message, conversation_window | sentiment, urgency_score, toxicity_flag | SentimentAnalysisTool |

### Tier 2: Knowledge & Decision Agents

| Agent | Purpose | Inputs | Outputs | Tools |
|-------|---------|--------|---------|-------|
| **RAGRetrievalAgent** | Retrieve knowledge from KB | query, intent, tenant_id | retrieved_passages, relevance_scores, source_ids | VectorSearchTool, MetadataFilterTool |
| **PolicyEvaluationAgent** | Apply business rules | intent, user_metadata, policy_rules | allowed_actions, denied_actions, conditions, reasoning | PolicyEngineTool, EligibilityCheckerTool, DateCalculatorTool |
| **ContextBuilderAgent** | Assemble structured context | intent, sentiment, rag_results, policy_results, user_profile | structured_context_bundle | None (aggregator) |

### Tier 3: Response & Action Agents

| Agent | Purpose | Inputs | Outputs | Tools |
|-------|---------|--------|---------|-------|
| **ResolutionGeneratorAgent** | Generate user-facing response | context_bundle, tone_config | final_answer, confidence, citations | None (LLM) |
| **EscalationDecisionAgent** | Decide on human escalation | confidence_scores, sentiment, policy_conflict | escalate, priority, reason | RuleEngineTool |
| **TicketActionAgent** | Create/update tickets | escalation_data, user_data | ticket_id, status, assigned_queue | CreateTicketTool, UpdateTicketTool, AssignQueueTool |

---

## Tool Architecture (18 Core Tools)

### Information Retrieval Tools

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| **VectorSearchTool** | Semantic search in vector DB | vector_collection, embedding_model |
| **MetadataFilterTool** | Filter search results by metadata | filter_fields, tenant_id |
| **FAQLookupTool** | Direct FAQ lookup | faq_index_path |
| **DocumentParserTool** | Parse documents for indexing | parser_config |

### Policy & Business Rules Tools

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| **PolicyEngineTool** | Execute business rule engine | policy_rules.yaml |
| **EligibilityCheckerTool** | Check user eligibility | eligibility_criteria |
| **DateCalculatorTool** | Calculate dates for windows/warranties | date_format, timezone |
| **RuleEngineTool** | Generic rule evaluation | rules_config |

### Customer Data Tools

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| **CustomerLookupTool** | Fetch customer profile | customer_db_connection |
| **OrderStatusTool** | Check order status | order_api_endpoint |
| **AccountLookupTool** | Get account details | account_db_connection |
| **HistoryLookupTool** | Get conversation history | history_db_connection |

### Ticket & Action Tools

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| **CreateTicketTool** | Create support ticket | ticket_api_endpoint |
| **UpdateTicketTool** | Update ticket status | ticket_api_endpoint |
| **AssignQueueTool** | Assign ticket to queue | queue_api_endpoint |
| **CloseTicketTool** | Close resolved ticket | ticket_api_endpoint |

### Communication Tools

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| **EmailTool** | Send email notifications | smtp_config |
| **SlackTool** | Send Slack notifications | webhook_url |
| **WebhookTool** | Generic webhook caller | webhook_url |

### Analysis Tools

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| **SentimentAnalysisTool** | Analyze message sentiment | sentiment_model |
| **ToxicityDetectorTool** | Detect toxic language | toxicity_threshold |

---

## Agent-to-Tool Permission Matrix

```
┌─────────────────────────────┬────────────────────────────────────────────────┐
│ Agent                        │ Allowed Tools                                  │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ IntentClassificationAgent   │ [None - LLM only]                              │
│ SentimentAgent              │ SentimentAnalysisTool, ToxicityDetectorTool    │
│ RAGRetrievalAgent           │ VectorSearchTool, MetadataFilterTool,          │
│                             │ FAQLookupTool, DocumentParserTool              │
│ PolicyEvaluationAgent       │ PolicyEngineTool, EligibilityCheckerTool,      │
│                             │ DateCalculatorTool, RuleEngineTool             │
│ ContextBuilderAgent         │ [None - aggregator only]                       │
│ ResolutionGeneratorAgent    │ [None - LLM only]                              │
│ EscalationDecisionAgent     │ RuleEngineTool, SentimentAnalysisTool         │
│ TicketActionAgent           │ CreateTicketTool, UpdateTicketTool,            │
│                             │ AssignQueueTool, CloseTicketTool               │
└─────────────────────────────┴────────────────────────────────────────────────┘
```

---

## Agent Execution Flow (Default Graph)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CUSTOMER MESSAGE INPUT                          │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. INTENT CLASSIFICATION AGENT                                     │
│     - Classifies user intent (refund, complaint, FAQ, etc.)         │
│     - Returns: intent, confidence, tool_mapping                     │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. SENTIMENT AGENT                                                 │
│     - Detects emotion (angry, neutral, positive)                    │
│     - Returns: sentiment, urgency_score, toxicity_flag              │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. RAG RETRIEVAL AGENT                                             │
│     - Searches knowledge base for relevant info                     │
│     - Returns: retrieved_passages, relevance_scores                 │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. POLICY EVALUATION AGENT                                         │
│     - Applies business rules (refund windows, warranties)           │
│     - Returns: allowed_actions, denied_actions, conditions          │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. CONTEXT BUILDER AGENT                                           │
│     - Assembles all previous outputs into structured context        │
│     - Returns: context_bundle                                       │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. RESOLUTION GENERATOR AGENT                                      │
│     - Generates final user-facing response                          │
│     - Returns: final_answer, confidence, citations                  │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7. ESCALATION DECISION AGENT                                       │
│     - Determines if human escalation needed                         │
│     - Returns: escalate (bool), priority, reason                    │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼ NO                           ▼ YES
            ┌───────────────┐           ┌───────────────────────┐
            │ RETURN TO USER│           │ 8. TICKET ACTION AGENT│
            │ WITH RESPONSE │           │ - Create ticket        │
            └───────────────┘           │ - Assign queue        │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │ RETURN TO USER        │
                                        │ WITH TICKET INFO      │
                                        └───────────────────────┘
```

---

## Conditional Execution Paths

### Early Escalation Path
```
IF SentimentAgent.urgency_score >= 0.9
   OR SentimentAgent.sentiment == "angry"
   OR ToxicityDetectorTool.toxicity == true
THEN
   Skip to EscalationDecisionAgent → TicketActionAgent
```

### FAQ Shortcut Path
```
IF IntentClassificationAgent.intent == "FAQ_QUERY"
   AND FAQLookupTool.finds_exact_match()
THEN
   Skip RAGRetrievalAgent, PolicyEvaluationAgent
   Go directly to ResolutionGeneratorAgent
```

### Direct Action Path
```
IF IntentClassificationAgent.tool_mapping == "order_status"
   OR IntentClassificationAgent.tool_mapping == "refund"
THEN
   Execute tool directly, skip policy evaluation
   Return result to user
```

---

## Configuration File Structure

### 1. Intents Configuration
**File**: `config/intents.json`
```json
{
  "intents": [
    {
      "label": "GREETING",
      "description": "User is greeting",
      "confidence_threshold": 0.7,
      "tool_mapping": null,
      "examples": ["Hello", "Hi"]
    },
    {
      "label": "FAQ_QUERY",
      "description": "User asking FAQ",
      "confidence_threshold": 0.7,
      "tool_mapping": "faq_lookup",
      "examples": ["What are your hours?"]
    },
    {
      "label": "REFUND_REQUEST",
      "description": "User wants refund",
      "confidence_threshold": 0.7,
      "tool_mapping": "refund",
      "examples": ["I want a refund"]
    },
    {
      "label": "COMPLAINT",
      "description": "User has complaint",
      "confidence_threshold": 0.7,
      "tool_mapping": "ticket_creation",
      "examples": ["I'm not happy"]
    }
  ]
}
```

### 2. Policy Configuration
**File**: `config/policies.json`
```json
{
  "policies": [
    {
      "intent_type": "REFUND_REQUEST",
      "business_action": "PROCESS_REFUND",
      "required_fields": ["order_id", "reason"],
      "conditions": {
        "max_refund_days": 30,
        "require_original_payment_method": true
      },
      "escalate_if_sentiment_above": 0.9,
      "priority": "high"
    }
  ]
}
```

### 3. Escalation Configuration
**File**: `config/escalation.json`
```json
{
  "enabled": true,
  "priority_thresholds": {
    "angry": "CRITICAL",
    "frustrated": "HIGH",
    "neutral": "NORMAL",
    "positive": "LOW"
  },
  "channels": {
    "ticket_system": {
      "description": "Create ticket in system",
      "tool_mapping": "create_ticket"
    },
    "email": {
      "description": "Send email notification",
      "tool_mapping": "email"
    }
  }
}
```

---

## State Schema (Graph State)

```python
{
  # Input state
  "user_message": str,
  "conversation_history": List[dict],
  "tenant_id": str,
  "session_id": str,
  "user_id": str,

  # Tier 1 outputs
  "intent": str,
  "intent_confidence": float,
  "intent_tool_mapping": Optional[str],

  "sentiment": str,
  "urgency_score": float,
  "toxicity_flag": bool,

  # Tier 2 outputs
  "retrieved_passages": List[dict],
  "retrieval_scores": List[float],
  "source_ids": List[str],

  "allowed_actions": List[str],
  "denied_actions": List[str],
  "policy_conditions": dict,
  "policy_reasoning": str,

  "context_bundle": dict,

  # Tier 3 outputs
  "final_response": str,
  "response_confidence": float,
  "citations": List[str],

  "should_escalate": bool,
  "escalation_priority": str,
  "escalation_reason": str,

  "ticket_id": Optional[str],
  "ticket_status": Optional[str],

  # Metadata
  "agent_trace": List[str],
  "tool_calls": List[dict],
  "errors": List[str]
}
```

---

## Multi-Tenant Configuration

### Tenant-Specific Overrides
```
config/
├── intents.json                    # Default intents
├── intents_amazon.json             # Amazon-specific intents
├── intents_flipkart.json           # Flipkart-specific intents
├── policies.json                   # Default policies
├── policies_amazon.json            # Amazon-specific policies
├── escalation.json                 # Default escalation
└── escalation_shopify.json         # Shopify-specific escalation
```

### Prompt Versioning
```
prompts/
├── IntentAgent/
│   ├── v1.txt                     # Default prompt
│   ├── v2.txt                     # Updated prompt
│   └── amazon/                    # Amazon-specific
│       └── v1.txt
├── SentimentAgent/
│   └── v1.txt
└── ResolutionGeneratorAgent/
    ├── v1.txt
    └── formal/                    # Formal tone variant
        └── v1.txt
```

---

## Tool Implementation Requirements

### Base Tool Interface
```python
class BaseTool:
    async def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """
        Execute tool with input payload.

        Returns:
            ToolResult with:
                - success: bool
                - data: Any
                - message: str
                - error: Optional[str]
        """
        pass
```

### Tool Configuration Schema
```python
{
    "tool_name": str,
    "tool_class": str,
    "description": str,
    "enabled": bool,
    "tenant_id": Optional[str],
    "config": {
        # Tool-specific configuration
    }
}
```

---

## Observability & Monitoring

### Metrics to Track
1. **Agent Metrics**
   - Execution time per agent
   - Confidence scores distribution
   - Error rates per agent

2. **Tool Metrics**
   - Tool call frequency
   - Tool success/failure rates
   - Tool execution latency

3. **Business Metrics**
   - Resolution rate
   - Escalation rate
   - Ticket creation rate
   - Customer satisfaction

### Logging Structure
```python
{
    "timestamp": str,
    "tenant_id": str,
    "session_id": str,
    "agent_name": str,
    "agent_input": dict,
    "agent_output": dict,
    "execution_time_ms": int,
    "tool_calls": [
        {
            "tool_name": str,
            "input": dict,
            "output": dict,
            "execution_time_ms": int
        }
    ]
}
```

---

## Security & Privacy

### API Key Management
- All API keys encrypted with AES-256-GCM
- Keys decrypted only at runtime
- Separate encryption per tenant

### Data Isolation
- Tenant-specific vector collections
- Tenant-specific database schemas
- No cross-tenant data access

### Tool Permissions
- Agent-to-tool permission matrix enforced
- No direct tool access bypassing agents
- Audit log for all tool executions

---

## Development Phases

### Phase 1: MVP (Minimal Viable Product)
- ✅ IntentClassificationAgent
- ✅ SentimentAgent
- ✅ RAGRetrievalAgent
- ✅ ResolutionGeneratorAgent
- ✅ EscalationDecisionAgent
- ✅ TicketActionAgent (basic)
- ✅ ToolRegistry with permissions

### Phase 2: Intelligence Layer
- PolicyEvaluationAgent
- ContextBuilderAgent
- Full policy engine
- Eligibility checking

### Phase 3: Production Hardening
- Multi-tenant isolation
- Observability & monitoring
- Performance optimization
- Comprehensive testing

### Phase 4: Advanced Features
- Answer quality evaluation
- Conversation summarization
- SLA prediction
- Multi-language support

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Total Agents** | 8 |
| **Total Tools** | 18 |
| **Understanding Agents** | 2 |
| **Knowledge/Decision Agents** | 3 |
| **Response/Action Agents** | 3 |
| **Information Tools** | 4 |
| **Policy Tools** | 4 |
| **Customer Data Tools** | 4 |
| **Ticket Tools** | 4 |
| **Communication Tools** | 3 |
| **Analysis Tools** | 2 |

---

## File Structure

```
customersupportresolution/
├── agents/                          # Agent implementations
│   ├── IntentAgent.py              ✅ Implemented
│   ├── SentimentAgent.py           ⏳ To implement
│   ├── RAGRetrievalAgent.py        ⏳ To implement
│   ├── PolicyEvaluationAgent.py    ⏳ To implement
│   ├── ContextBuilderAgent.py      ⏳ To implement
│   ├── ResolutionGeneratorAgent.py ⏳ To implement
│   ├── EscalationDecisionAgent.py  ⏳ To implement
│   └── TicketActionAgent.py        ⏳ To implement
├── tools/                           # Tool implementations
│   ├── information/                 # Information retrieval
│   │   ├── VectorSearchTool.py
│   │   ├── MetadataFilterTool.py
│   │   ├── FAQLookupTool.py
│   │   └── DocumentParserTool.py
│   ├── policy/                      # Policy & rules
│   │   ├── PolicyEngineTool.py
│   │   ├── EligibilityCheckerTool.py
│   │   ├── DateCalculatorTool.py
│   │   └── RuleEngineTool.py
│   ├── customer/                    # Customer data
│   │   ├── CustomerLookupTool.py
│   │   ├── OrderStatusTool.py
│   │   ├── AccountLookupTool.py
│   │   └── HistoryLookupTool.py
│   ├── ticket/                      # Ticket actions
│   │   ├── CreateTicketTool.py
│   │   ├── UpdateTicketTool.py
│   │   ├── AssignQueueTool.py
│   │   └── CloseTicketTool.py
│   ├── communication/               # Communications
│   │   ├── EmailTool.py
│   │   ├── SlackTool.py
│   │   └── WebhookTool.py
│   └── analysis/                    # Analysis
│       ├── SentimentAnalysisTool.py
│       └── ToxicityDetectorTool.py
├── config/                          # Configuration files
│   ├── intents.json                ✅ Created
│   ├── policies.json               ⏳ To create
│   ├── escalation.json             ⏳ To create
│   └── tenants/                    # Tenant-specific configs
├── prompts/                         # Prompt templates
│   ├── IntentAgent/                ✅ Created
│   ├── SentimentAgent/
│   ├── RAGRetrievalAgent/
│   ├── PolicyEvaluationAgent/
│   ├── ContextBuilderAgent/
│   ├── ResolutionGeneratorAgent/
│   ├── EscalationDecisionAgent/
│   └── TicketActionAgent/
├── core/                            # Core framework
│   ├── BaseAgent.py                ✅ Implemented
│   ├── ToolRegistry.py             ✅ Implemented
│   ├── GraphEngine.py              ✅ Implemented
│   ├── GraphRouting.py             ✅ Implemented
│   └── GraphNodes.py               ✅ Implemented
├── services/                        # Services
│   ├── ChatService.py              ✅ Implemented
│   ├── ConfigService.py            ✅ Implemented
│   └── ResponseBuilder.py          ✅ Implemented
└── routes/                          # API endpoints
    ├── chat.py                     ✅ Implemented
    └── admin.py                    ⏳ To create
```

---

## Next Steps

1. **Implement remaining agents** (7 agents remaining)
2. **Create tool implementations** (18 tools)
3. **Create policy and escalation config files**
4. **Implement prompt templates for all agents**
5. **Set up multi-tenant configuration**
6. **Implement observability and monitoring**
7. **Create admin API for configuration management**

---

**Document Version**: 1.0
**Last Updated**: 2026-03-17
**Status**: IntentAgent ✅ Complete | Rest ⏳ In Progress
