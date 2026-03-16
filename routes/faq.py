"""FAQ management routes for CRUD operations on Qdrant vector store.

Endpoints:
- POST /faq/create - Create single FAQ
- POST /faq/bulk - Bulk create FAQs via CSV/XLSX
- GET /faq/{tenant_id} - Get all FAQs by tenant
- PUT /faq/{faq_id} - Update FAQ
- DELETE /faq/{tenant_id}/{faq_id} - Delete FAQ
- DELETE /faq/{tenant_id}/all - Delete all FAQs for tenant
"""

import io
import csv
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.FAQService import FAQService, FAQItem
from services.BootstrapService import bootstrap_system
from utils.Config import CONFIG
from utils.Embeddings import HuggingFaceEmbeddings
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faq", tags=["FAQ"])

# Initialize container and embedding model
container = bootstrap_system(CONFIG)
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("BGE_BASE", "BAAI/bge-base-en-v1.5")
)


# Pydantic models for request/response
class FAQCreateRequest(BaseModel):
    tenant_id: str
    category: str
    question: str
    answer: str


class FAQUpdateRequest(BaseModel):
    tenant_id: str
    category: str
    question: str
    answer: str


class FAQResponse(BaseModel):
    id: Optional[str] = None
    tenant_id: str
    category: str
    question: str
    answer: str


def get_faq_service() -> FAQService:
    """Get FAQ service instance."""
    return FAQService(
        vector_store=container.vector_store.vector_store,
        embedding_model=embedding_model
    )


@router.post("/create", response_model=FAQResponse)
async def create_faq(request: FAQCreateRequest):
    """Create a single FAQ.

    Request body:
    {
        "tenant_id": "amazon",
        "category": "shipping",
        "question": "Where is my Prime delivery?",
        "answer": "Prime deliveries arrive within 1-2 business days."
    }
    """
    try:
        service = get_faq_service()

        faq = FAQItem(
            tenant_id=request.tenant_id,
            category=request.category,
            question=request.question,
            answer=request.answer
        )

        result = service.add_faq(faq)

        return FAQResponse(**result)

    except Exception as e:
        logger.error(f"Error creating FAQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk")
async def bulk_create_faqs(
    tenant_id: str = Query(..., description="Tenant ID"),
    file: UploadFile = File(..., description="CSV or XLSX file with columns: question,answer,category")
):
    """Bulk create FAQs from CSV or XLSX file.

    File format:
    - CSV: question,answer,category (comma-separated)
    - XLSX: same column names

    Example CSV:
    question,answer,category
    "Where is my order?","Track your order in My Orders section","shipping"
    "How to return?","Go to My Orders > Select item > Return","returns"
    """
    try:
        service = get_faq_service()

        # Read file based on type
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            # Read CSV
            content = await file.read()
            csv_reader = csv.DictReader(io.StringIO(content.decode('utf-8')))

            faqs = []
            for row in csv_reader:
                # Support both comma and array in category
                category = row.get('category', 'general')
                if isinstance(category, str) and ',' in category:
                    category = category.split(',')[0].strip()

                faqs.append(FAQItem(
                    tenant_id=tenant_id,
                    category=category,
                    question=row['question'].strip(),
                    answer=row['answer'].strip()
                ))

        elif filename.endswith('.xlsx'):
            # Read XLSX
            import openpyxl

            content = await file.read()
            workbook = openpyxl.load_workbook(io.BytesIO(content))
            sheet = workbook.active

            # Get header row
            headers = [cell.value for cell in sheet[1]]
            headers = [h.lower() if h else "" for h in headers]

            # Find column indices
            question_idx = headers.index('question') if 'question' in headers else 0
            answer_idx = headers.index('answer') if 'answer' in headers else 1
            category_idx = headers.index('category') if 'category' in headers else 2

            faqs = []
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if not row[question_idx]:
                    continue

                category = row[category_idx] if category_idx < len(row) else 'general'
                if isinstance(category, str) and ',' in category:
                    category = category.split(',')[0].strip()

                faqs.append(FAQItem(
                    tenant_id=tenant_id,
                    category=category or 'general',
                    question=str(row[question_idx]).strip(),
                    answer=str(row[answer_idx]).strip()
                ))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use .csv or .xlsx"
            )

        if not faqs:
            raise HTTPException(status_code=400, detail="No valid FAQs found in file")

        # Add bulk FAQs
        result = service.add_bulk_faqs(faqs)

        return JSONResponse(content={
            "success": result["success"],
            "created": result["created"],
            "failed": result["failed"],
            "details": result["details"]
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk creating FAQs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tenant_id}", response_model=List[FAQResponse])
async def get_all_faqs(tenant_id: str):
    """Get all FAQs for a tenant.

    Returns list of FAQs with their IDs.
    """
    try:
        service = get_faq_service()
        faqs = service.get_all_faqs(tenant_id)

        return [FAQResponse(**faq) for faq in faqs]

    except Exception as e:
        logger.error(f"Error getting FAQs for tenant '{tenant_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tenant_id}/{faq_id}", response_model=FAQResponse)
async def get_faq_by_id(tenant_id: str, faq_id: str):
    """Get a specific FAQ by ID.

    Returns the FAQ with the given ID.
    """
    try:
        service = get_faq_service()
        faqs = service.get_all_faqs(tenant_id)

        for faq in faqs:
            if faq.get("id") == faq_id:
                return FAQResponse(**faq)

        raise HTTPException(status_code=404, detail="FAQ not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting FAQ {faq_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{faq_id}", response_model=FAQResponse)
async def update_faq(faq_id: str, request: FAQUpdateRequest):
    """Update an existing FAQ.

    Request body:
    {
        "tenant_id": "amazon",
        "category": "shipping",
        "question": "Where is my Prime delivery?",
        "answer": "Updated answer here"
    }
    """
    try:
        service = get_faq_service()

        faq = FAQItem(
            tenant_id=request.tenant_id,
            category=request.category,
            question=request.question,
            answer=request.answer
        )

        result = service.update_faq(faq_id, faq)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "FAQ not found"))

        return FAQResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating FAQ {faq_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tenant_id}/{faq_id}")
async def delete_faq(tenant_id: str, faq_id: str):
    """Delete a specific FAQ by ID.

    Returns deletion status.
    """
    try:
        service = get_faq_service()
        result = service.delete_faq(tenant_id, faq_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "FAQ not found"))

        return {"success": True, "id": faq_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting FAQ {faq_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tenant_id}/all")
async def delete_all_faqs(tenant_id: str):
    """Delete all FAQs for a tenant.

    WARNING: This deletes all FAQs in the tenant's collection.
    """
    try:
        service = get_faq_service()
        result = service.delete_all_faqs(tenant_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Collection not found"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting all FAQs for tenant '{tenant_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tenant_id}/stats")
async def get_faq_stats(tenant_id: str):
    """Get statistics for tenant's FAQ collection.

    Returns count, vector size, etc.
    """
    try:
        collection_name = f"{tenant_id}_faqs"

        stats = container.vector_store.vector_store.get_collection_stats(collection_name)

        if not stats.get("exists"):
            raise HTTPException(status_code=404, detail="FAQ collection not found")

        return {
            "tenant_id": tenant_id,
            "collection_name": collection_name,
            "total_faqs": stats.get("count", 0),
            "vector_size": stats.get("vector_size")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting FAQ stats for tenant '{tenant_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
