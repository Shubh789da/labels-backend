"""API routes for drug approval history endpoints."""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from pydantic import BaseModel, HttpUrl

from app.models import (
    DrugHistoryResponse,
    DrugApproval,
    DrugSearchResult,
)
from app.services import DrugHistoryAggregator, RxNormService, OpenFDAService, OCRServiceClient, get_indication_pipeline

router = APIRouter(prefix="/drugs", tags=["drugs"])


# OCR Request/Response Models
class DocumentOCRRequest(BaseModel):
    """Request to process an FDA document with OCR."""
    url: HttpUrl
    document_id: Optional[str] = None
    document_type: Optional[str] = None


class DocumentOCRResponse(BaseModel):
    """Response from document OCR processing."""
    success: bool
    document_id: Optional[str]
    s3_url: Optional[str]
    page_count: int = 0
    message: str


def get_aggregator() -> DrugHistoryAggregator:
    """Dependency to get aggregator service."""
    return DrugHistoryAggregator()


def get_rxnorm_service() -> RxNormService:
    """Dependency to get RxNorm service."""
    return RxNormService()


def get_openfda_service() -> OpenFDAService:
    """Dependency to get openFDA service."""
    return OpenFDAService()


@router.get(
    "/search",
    response_model=list[DrugSearchResult],
    summary="Search for drugs by name",
    description="Search for drugs using RxNorm to get standardized drug identifiers.",
)
async def search_drugs(
    name: str = Query(..., min_length=2, description="Drug name to search for"),
    rxnorm_service: RxNormService = Depends(get_rxnorm_service),
) -> list[DrugSearchResult]:
    """Search for drugs by name using RxNorm.

    Returns a list of matching drugs with their RxCUI identifiers,
    which can be used for more precise queries.
    """
    try:
        results = await rxnorm_service.search_drugs(name)

        if not results:
            # Try approximate matching
            results = await rxnorm_service.get_approximate_match(name)

        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching for drugs: {str(e)}",
        )
    finally:
        await rxnorm_service.close()


@router.get(
    "/history/{drug_name}",
    response_model=DrugHistoryResponse,
    summary="Get drug approval history",
    description="Get comprehensive drug approval history from FDA and other public sources.",
)
async def get_drug_history(
    drug_name: str,
    include_dailymed: bool = Query(
        True,
        description="Include DailyMed data (may increase response time)",
    ),
    aggregator: DrugHistoryAggregator = Depends(get_aggregator),
) -> DrugHistoryResponse:
    """Get comprehensive drug approval history.

    This endpoint aggregates data from multiple sources:
    - openFDA Drugs@FDA: FDA approval records and submissions
    - openFDA Drug Labels: Indication information from drug labeling
    - DailyMed: Structured Product Labeling (SPL) data
    - RxNorm: Drug name normalization

    The response includes:
    - Approval records with submission history
    - Therapeutic indications
    - Chronological timeline of approval events
    """
    try:
        result = await aggregator.get_drug_history(
            drug_name,
            include_dailymed=include_dailymed,
        )

        if not result.approvals and not result.indications:
            raise HTTPException(
                status_code=404,
                detail=f"No approval history found for drug: {drug_name}",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving drug history: {str(e)}",
        )
    finally:
        await aggregator.close()


@router.get(
    "/application/{application_number}",
    response_model=DrugApproval,
    summary="Get drug by FDA application number",
    description="Get drug approval details by FDA application number (NDA, ANDA, or BLA).",
)
async def get_drug_by_application(
    application_number: str,
    openfda_service: OpenFDAService = Depends(get_openfda_service),
) -> DrugApproval:
    """Get drug approval details by FDA application number.

    Application numbers follow these formats:
    - NDA: New Drug Application (e.g., NDA012345)
    - ANDA: Abbreviated New Drug Application (e.g., ANDA012345)
    - BLA: Biologics License Application (e.g., BLA123456)
    """
    try:
        result = await openfda_service.get_drug_by_application_number(application_number)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No drug found for application number: {application_number}",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving drug by application: {str(e)}",
        )
    finally:
        await openfda_service.close()


@router.get(
    "/approvals",
    response_model=list[DrugApproval],
    summary="Search FDA drug approvals",
    description="Search FDA drug approvals database by drug name.",
)
async def search_approvals(
    drug_name: str = Query(..., min_length=2, description="Drug name to search"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    openfda_service: OpenFDAService = Depends(get_openfda_service),
) -> list[DrugApproval]:
    """Search FDA drug approvals by drug name.

    Returns approval records from the Drugs@FDA database including:
    - Application numbers
    - Sponsor information
    - Submission history with dates
    - Product formulations
    """
    try:
        results = await openfda_service.search_drug_approvals(drug_name, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching approvals: {str(e)}",
        )
    finally:
        await openfda_service.close()


@router.get(
    "/indications/{drug_name}",
    summary="Get drug indications",
    description="Get therapeutic indications for a drug from FDA labeling.",
)
async def get_drug_indications(
    drug_name: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    openfda_service: OpenFDAService = Depends(get_openfda_service),
):
    """Get therapeutic indications for a drug.

    Extracts indication information from FDA drug labeling data.
    Indications describe the approved uses and conditions the drug treats.
    """
    try:
        indications = await openfda_service.search_drug_labels(drug_name, limit=limit)

        if not indications:
            raise HTTPException(
                status_code=404,
                detail=f"No indications found for drug: {drug_name}",
            )

        return {
            "drug_name": drug_name,
            "indications": indications,
            "count": len(indications),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving indications: {str(e)}",
        )
    finally:
        await openfda_service.close()


@router.get(
    "/normalize/{drug_name}",
    summary="Normalize drug name",
    description="Get the standardized drug name from RxNorm.",
)
async def normalize_drug_name(
    drug_name: str,
    rxnorm_service: RxNormService = Depends(get_rxnorm_service),
):
    """Normalize a drug name using RxNorm.

    Returns the standardized drug name, RxCUI, and term type.
    Useful for ensuring consistent drug identification across queries.
    """
    try:
        result = await rxnorm_service.normalize_drug_name(drug_name)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Could not normalize drug name: {drug_name}",
            )

        return {
            "original_name": drug_name,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error normalizing drug name: {str(e)}",
        )
    finally:
        await rxnorm_service.close()


# OCR Endpoints
def get_ocr_service() -> OCRServiceClient:
    """Dependency to get OCR service client."""
    return OCRServiceClient()


@router.post(
    "/ocr/process",
    response_model=DocumentOCRResponse,
    summary="Process FDA document with OCR",
    description="Send an FDA document PDF to DeepSeek-OCR-2 for text extraction.",
    tags=["ocr"],
)
async def process_document_ocr(
    request: DocumentOCRRequest,
    ocr_service: OCRServiceClient = Depends(get_ocr_service),
) -> DocumentOCRResponse:
    """Process an FDA document PDF with DeepSeek-OCR-2.

    Sends the PDF URL to the OCR service running on RunPod.
    The service extracts text from all pages, converts to markdown,
    and uploads to S3.

    Returns the S3 URL where the markdown file can be downloaded.
    """
    try:
        result = await ocr_service.process_document(
            pdf_url=str(request.url),
            document_id=request.document_id,
            document_type=request.document_type,
        )

        if not result:
            raise HTTPException(
                status_code=503,
                detail="OCR service unavailable or failed to process document",
            )

        return DocumentOCRResponse(
            success=result.get("success", False),
            document_id=result.get("document_id"),
            s3_url=result.get("s3_url"),
            page_count=result.get("page_count", 0),
            message=result.get("message", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )
    finally:
        await ocr_service.close()


@router.get(
    "/ocr/health",
    summary="Check OCR service health",
    description="Check if the DeepSeek-OCR-2 service on RunPod is available.",
    tags=["ocr"],
)
async def check_ocr_health(
    ocr_service: OCRServiceClient = Depends(get_ocr_service),
):
    """Check if the OCR service is healthy and available."""
    try:
        is_healthy = await ocr_service.health_check()
        return {
            "status": "healthy" if is_healthy else "unavailable",
            "service": "DeepSeek-OCR-2",
        }
    finally:
        await ocr_service.close()


class IndicationExtractionRequest(BaseModel):
    """Request to extract indications from FDA label PDF."""
    pdf_url: HttpUrl
    filename_prefix: str


class IndicationExtractionResponse(BaseModel):
    """Response from indication extraction pipeline."""
    found: bool
    text: str
    method: str  # "local", "ocr", or "failed"
    formatted: bool
    indication_count: int = 0
    error_message: Optional[str] = None


@router.post(
    "/extract-indication",
    response_model=IndicationExtractionResponse,
    summary="Extract indication from FDA Label",
    description="Extracts and formats the 'Indications and Usage' section from an FDA label PDF. Uses local extraction first, falls back to OCR if needed.",
)
async def extract_indication(
    request: IndicationExtractionRequest,
):
    """
    Extract and format indication text from a PDF URL.
    
    Pipeline:
    1. Attempts local PDF text extraction using PyMuPDF
    2. Falls back to RunPod OCR if local extraction fails
    3. Formats extracted text into bullet points using DeepSeek LLM
    
    Returns formatted bullet points for frontend display.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"[extract-indication] Request for PDF: {request.pdf_url}")
    logger.info(f"[extract-indication] Filename prefix: {request.filename_prefix}")

    pipeline = get_indication_pipeline()
    
    try:
        result = await pipeline.extract_indications(
            pdf_url=str(request.pdf_url),
            filename_prefix=request.filename_prefix
        )

        if not result.found:
            logger.warning(f"[extract-indication] Extraction failed for {request.filename_prefix}")
        else:
            logger.info(f"[extract-indication] Successfully extracted via {result.method.value} for {request.filename_prefix}")

        return IndicationExtractionResponse(
            found=result.found,
            text=result.text,
            method=result.method.value,
            formatted=result.formatted,
            indication_count=result.indication_count,
            error_message=result.error_message
        )
    except Exception as e:
        logger.error(f"[extract-indication] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await pipeline.close()
