# RAG Training Test

## Files: PDF, MD, TXT

### Knowledge Base Training

```bash
# Upload file
curl -X POST "http://localhost:8000/training/" \
  -F "file=@data/knowledge/amazon_faq.md" \
  -F "tenant_id=default"
```

#### Response
```json
{"status": "success", "collection": "knowledge_base_default", "chunks_added": 45}
```

---

## FAQ Training

```bash
# Upload FAQ file
curl -X POST "http://localhost:8000/faq/" \
  -F "file=@data/faq/shopify/faq.md" \
  -F "tenant_id=shopify"
```

### Response
```json
{"status": "success", "collection": "faq_shopify", "chunks_added": 25}
```

---

## Collection Info

```bash
# Get FAQ collection info
curl "http://localhost:8000/faq/info?tenant_id=shopify"
```

---

## Collection Naming

- Knowledge Base: `knowledge_base_{tenant_id}`
- FAQ: `faq_{tenant_id}`
