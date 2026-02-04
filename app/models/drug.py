"""Pydantic models for drug approval history data."""
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class DataSource(str, Enum):
    """Enumeration of data sources."""
    OPENFDA_DRUGSFDA = "openfda_drugsfda"
    OPENFDA_LABEL = "openfda_label"
    DAILYMED = "dailymed"
    RXNORM = "rxnorm"


class ApplicationDocument(BaseModel):
    """Represents an FDA application document (label, letter, review, etc.)."""
    id: Optional[str] = Field(None, description="Document ID")
    url: Optional[str] = Field(None, description="URL to the document (PDF)")
    date: Optional[str] = Field(None, description="Document date (YYYYMMDD format)")
    type: Optional[str] = Field(None, description="Document type (e.g., Label, Letter, Review)")


class DrugSubmission(BaseModel):
    """Represents a regulatory submission for a drug."""
    submission_type: Optional[str] = Field(None, description="Type of submission (e.g., ORIG, SUPPL)")
    submission_number: Optional[str] = Field(None, description="Submission number")
    submission_status: Optional[str] = Field(None, description="Status of submission (e.g., AP for approved)")
    submission_status_date: Optional[str] = Field(None, description="Date of submission status")
    submission_class_code: Optional[str] = Field(None, description="Submission class code")
    submission_class_code_description: Optional[str] = Field(None, description="Description of submission class")
    review_priority: Optional[str] = Field(None, description="Review priority (e.g., STANDARD, PRIORITY)")
    application_docs: list[ApplicationDocument] = Field(default_factory=list, description="Associated documents (labels, letters, reviews)")


class DrugIndication(BaseModel):
    """Represents a therapeutic indication for a drug."""
    indication_text: str = Field(..., description="Description of the indication")
    approval_date: Optional[str] = Field(None, description="Date when this indication was approved")
    source: DataSource = Field(..., description="Source of this indication data")
    url: Optional[str] = Field(None, description="URL to the source document (e.g., DailyMed)")
    is_original: bool = Field(False, description="Whether this is the original/first approved indication")


class DrugProduct(BaseModel):
    """Represents a specific drug product formulation."""
    brand_name: Optional[str] = Field(None, description="Brand name of the product")
    generic_name: Optional[str] = Field(None, description="Generic/active ingredient name")
    dosage_form: Optional[str] = Field(None, description="Form of the drug (e.g., TABLET, CAPSULE)")
    route: Optional[str] = Field(None, description="Route of administration")
    strength: Optional[str] = Field(None, description="Drug strength")
    marketing_status: Optional[str] = Field(None, description="Marketing status")
    active_ingredients: list[str] = Field(default_factory=list, description="List of active ingredients")


class DrugApproval(BaseModel):
    """Represents a drug approval record from FDA."""
    application_number: str = Field(..., description="FDA application number (NDA/ANDA/BLA)")
    sponsor_name: Optional[str] = Field(None, description="Company that sponsored the application")
    application_type: Optional[str] = Field(None, description="Type: NDA, ANDA, or BLA")
    submissions: list[DrugSubmission] = Field(default_factory=list, description="List of submissions")
    products: list[DrugProduct] = Field(default_factory=list, description="Associated products")
    initial_approval_date: Optional[str] = Field(None, description="Date of initial approval")
    source: DataSource = Field(DataSource.OPENFDA_DRUGSFDA, description="Data source")


class DrugSearchResult(BaseModel):
    """Result from searching for a drug by name."""
    rxcui: Optional[str] = Field(None, description="RxNorm Concept Unique Identifier")
    name: str = Field(..., description="Drug name")
    synonym: Optional[str] = Field(None, description="Alternative name/synonym")
    tty: Optional[str] = Field(None, description="Term type")


class DrugHistoryResponse(BaseModel):
    """Complete drug approval history response."""
    drug_name: str = Field(..., description="Searched drug name")
    normalized_name: Optional[str] = Field(None, description="Normalized drug name from RxNorm")
    rxcui: Optional[str] = Field(None, description="RxNorm Concept Unique Identifier")
    approvals: list[DrugApproval] = Field(default_factory=list, description="FDA approval records")
    indications: list[DrugIndication] = Field(default_factory=list, description="Therapeutic indications")
    indication_count: Optional[int] = Field(0, description="Count of approved indications analyzed by AI")
    key_documents: list[dict] = Field(default_factory=list, description="Key documents (First Label, Latest Label)")
    timeline: list[dict] = Field(default_factory=list, description="Chronological timeline of approvals")
    sources_queried: list[str] = Field(default_factory=list, description="Data sources that were queried")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered during data retrieval")
