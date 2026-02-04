"""Service for aggregating drug history data from multiple sources."""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from app.services.openfda_service import OpenFDAService
from app.services.dailymed_service import DailyMedService
from app.services.rxnorm_service import RxNormService
from app.models import DrugHistoryResponse, DrugApproval, DrugIndication
from app.services.ocr_client import OCRServiceClient

logger = logging.getLogger(__name__)


class DrugHistoryAggregator:
    """Aggregates drug approval history from multiple public data sources.

    Data Sources:
    - openFDA Drugs@FDA: FDA approval records, submissions, application numbers
    - openFDA Drug Labels: Indication text from drug labeling
    - DailyMed: SPL (Structured Product Labeling) data
    - RxNorm: Drug name normalization and identification
    """

    def __init__(self):
        self.openfda_service = OpenFDAService()
        self.dailymed_service = DailyMedService()
        self.openfda_service = OpenFDAService()
        self.dailymed_service = DailyMedService()
        self.rxnorm_service = RxNormService()
        self.ocr_client = OCRServiceClient()

    async def close(self):
        """Close all service connections."""
        await asyncio.gather(
            self.openfda_service.close(),
            self.dailymed_service.close(),
            self.rxnorm_service.close(),
        )

    async def get_drug_history(
        self,
        drug_name: str,
        include_dailymed: bool = True,
    ) -> DrugHistoryResponse:
        """Get comprehensive drug approval history from all sources.

        Args:
            drug_name: Name of the drug to search for
            include_dailymed: Whether to include DailyMed data (slower)

        Returns:
            Aggregated drug history response
        """
        errors = []
        sources_queried = []

        # Step 1: Normalize drug name using RxNorm
        normalized_info = None
        try:
            normalized_info = await self.rxnorm_service.normalize_drug_name(drug_name)
            sources_queried.append("RxNorm")
        except Exception as e:
            logger.error(f"RxNorm error for {drug_name}: {e}")
            errors.append(f"RxNorm lookup failed: {str(e)}")

        # Use normalized name if available, otherwise use original
        search_name = drug_name
        if normalized_info and normalized_info.get("normalized_name"):
            search_name = normalized_info["normalized_name"]

        # Step 2: Fetch data from all sources concurrently
        tasks = [
            self._fetch_openfda_approvals(search_name, drug_name),
            self._fetch_openfda_indications(search_name, drug_name),
        ]

        if include_dailymed:
            tasks.append(self._fetch_dailymed_data(search_name, drug_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        approvals = []
        indications = []

        # openFDA approvals
        if isinstance(results[0], tuple):
            approvals_data, source, error = results[0]
            if approvals_data:
                approvals.extend(approvals_data)
            if source:
                sources_queried.append(source)
            if error:
                errors.append(error)

        # openFDA indications
        if isinstance(results[1], tuple):
            indications_data, source, error = results[1]
            if indications_data:
                indications.extend(indications_data)
            if source:
                sources_queried.append(source)
            if error:
                errors.append(error)

        # DailyMed data
        if include_dailymed and len(results) > 2 and isinstance(results[2], tuple):
            dailymed_indications, source, error = results[2]
            if dailymed_indications:
                indications.extend(dailymed_indications)
            if source:
                sources_queried.append(source)
            if error:
                errors.append(error)

        # Step 3: Build timeline
        timeline = self._build_timeline(approvals, indications)

        # Step 4: Deduplicate indications
        indications = self._deduplicate_indications(indications)
        
        # Step 5: AI Analysis - Count Indications
        indication_count = 0
        try:
            # Find the best indication text to analyze (longest one usually has the full list)
            best_text_idx = -1
            max_len = 0
            
            for i, ind in enumerate(indications):
                if len(ind.indication_text) > max_len:
                    max_len = len(ind.indication_text)
                    best_text_idx = i
            
            if best_text_idx >= 0:
                from app.services.deepseek_llm import get_deepseek_llm_service
                llm_service = get_deepseek_llm_service()
                indication_count = await llm_service.count_indications(indications[best_text_idx].indication_text)
        except Exception as e:
            logger.error(f"Error counting indications: {e}")
            # Fallback to simple list count
            indication_count = len(indications)

        # Step 6: Identify Key Documents (First Approval & Latest Label)
        key_documents = []
        
        # Filter events with URLs
        doc_events = [e for e in timeline if e.get("url")]
        
        if doc_events:
            # Sort chronologically
            doc_events.sort(key=lambda x: x["date"])
            
            # First Document (Earliest)
            first_doc = doc_events[-1] # Default to something
            
            # Try to find specific "Initial Approval" or earliest label
            for event in reversed(doc_events): # Reversed = Oldest to Newest since sorted reversed? No, string sort "2002" < "2024". 
                # Wait, build_timeline produces default order? 
                # doc_events sorted by string date ascending: index 0 is oldest ("2002..."), index -1 is newest ("2024...")
                pass
                
            # Actually let's just grab Oldest and Newest
            first_doc = doc_events[0]   # Oldest
            latest_doc = doc_events[-1] # Newest
            
            # Format for frontend
            key_documents.append({
                "title": "First Approval Label",
                "date": first_doc["date"],
                "url": first_doc["url"],
                "type": "initial"
            })
            
            # Only add latest if it's different
            if latest_doc["url"] != first_doc["url"]:
                key_documents.append({
                    "title": "Latest Label",
                    "date": latest_doc["date"],
                    "url": latest_doc["url"],
                    "type": "latest"
                })

        return DrugHistoryResponse(
            drug_name=drug_name,
            normalized_name=normalized_info.get("normalized_name") if normalized_info else None,
            rxcui=normalized_info.get("rxcui") if normalized_info else None,
            approvals=approvals,
            indications=indications,
            indication_count=indication_count,
            key_documents=key_documents,
            timeline=timeline,
            sources_queried=list(set(sources_queried)),
            errors=errors,
        )

    async def _fetch_openfda_approvals(
        self,
        search_name: str,
        original_name: str,
    ) -> tuple[list[DrugApproval], Optional[str], Optional[str]]:
        """Fetch drug approvals from openFDA."""
        try:
            approvals = await self.openfda_service.search_drug_approvals(search_name)

            # If no results with normalized name, try original
            if not approvals and search_name != original_name:
                approvals = await self.openfda_service.search_drug_approvals(
                    original_name
                )

            return approvals, "openFDA Drugs@FDA", None
        except Exception as e:
            logger.error(f"openFDA approvals error: {e}")
            return [], "openFDA Drugs@FDA", f"openFDA approvals failed: {str(e)}"

    async def _fetch_openfda_indications(
        self,
        search_name: str,
        original_name: str,
    ) -> tuple[list[DrugIndication], Optional[str], Optional[str]]:
        """Fetch drug indications from openFDA labels."""
        try:
            indications = await self.openfda_service.search_drug_labels(search_name)

            # If no results with normalized name, try original
            if not indications and search_name != original_name:
                indications = await self.openfda_service.search_drug_labels(
                    original_name
                )

            return indications, "openFDA Drug Labels", None
        except Exception as e:
            logger.error(f"openFDA labels error: {e}")
            return [], "openFDA Drug Labels", f"openFDA labels failed: {str(e)}"

    async def _fetch_dailymed_data(
        self,
        search_name: str,
        original_name: str,
    ) -> tuple[list[DrugIndication], Optional[str], Optional[str]]:
        """Fetch drug data from DailyMed."""
        try:
            indications = await self.dailymed_service.get_drug_indications(search_name)

            # If no results with normalized name, try original
            if not indications and search_name != original_name:
                indications = await self.dailymed_service.get_drug_indications(
                    original_name
                )

            return indications, "DailyMed", None
        except Exception as e:
            logger.error(f"DailyMed error: {e}")
            return [], "DailyMed", f"DailyMed failed: {str(e)}"

    def _build_timeline(
        self,
        approvals: list[DrugApproval],
        indications: list[DrugIndication],
    ) -> list[dict]:
        """Build a chronological timeline of drug approval events.

        Args:
            approvals: List of drug approvals
            indications: List of indications

        Returns:
            List of timeline events sorted by date
        """
        events = []

        # Add approval events from submissions
        for approval in approvals:
            for submission in approval.submissions:
                if submission.submission_status_date:
                    # Filter by submission class code
                    # Criteria:
                    # 1. Always allow "ORIG" (Original) submissions.
                    # 2. For others (e.g. SUPPL), only allow if Class Code or Description contains:
                    #    - LABELING
                    #    - EFFICACY
                    #    - TYPE 3
                    
                    is_orig = submission.submission_type == "ORIG"
                    has_valid_code = False
                    
                    if submission.submission_class_code:
                        code_full = f"{submission.submission_class_code} {submission.submission_class_code_description or ''}".upper()
                        allowed_terms = ["LABELING", "EFFICACY", "TYPE 3"]
                        if any(term in code_full for term in allowed_terms):
                            has_valid_code = True
                            
                    # Explicitly exclude if not satisfying above
                    if not (is_orig or has_valid_code):
                        continue

                    # Find best document URL (prefer Label, then Letter)
                    doc_url = None
                    label_url = None
                    letter_url = None
                    
                    for doc in submission.application_docs:
                        if doc.url and doc.type:
                            if "Label" in doc.type:
                                label_url = doc.url
                            elif "Letter" in doc.type:
                                letter_url = doc.url
                    
                    # Use Label if available, otherwise Letter
                    if label_url:
                        doc_url = label_url
                    elif letter_url:
                        doc_url = letter_url
                    elif submission.application_docs and submission.application_docs[0].url:
                         doc_url = submission.application_docs[0].url
                    
                    
                    # Check if we need to extract indications (only if we have a label)
                    needs_ocr = False
                    if label_url:
                        needs_ocr = True

                    # Determine description
                    description = self._get_event_description(approval, submission)
                    
                    # If we need OCR, we modify the description or metadata to indicate loading
                    # Per user request: "make 1 event... description the indication and usage will go"
                    # So initially we show placeholder
                    
                    is_loading = False
                    filename_prefix = None
                    
                    if needs_ocr:
                        description = "Reading PDF to fetch the indication..."
                        is_loading = True
                        filename_prefix = f"{approval.application_number}_{submission.submission_number}"

                    event = {
                        "date": submission.submission_status_date,
                        "event_type": self._get_event_type(submission),
                        "description": description,
                        "application_number": approval.application_number,
                        "sponsor": approval.sponsor_name,
                        "source": approval.source.value,
                        "url": doc_url,
                        "is_loading": is_loading,
                        "filename_prefix": filename_prefix
                    }
                    events.append(event)
                    
                    # We no longer create a separate indication event for the OCR placeholder
                    # The main event serves as the container

        # Add indication events with dates
        for indication in indications:
            if indication.approval_date:
                event = {
                    "date": indication.approval_date,
                    "event_type": "indication",
                    "description": indication.indication_text[:200] + "..."
                    if len(indication.indication_text) > 200
                    else indication.indication_text,
                    "source": indication.source.value,
                    "url": indication.url,
                }
                events.append(event)

        return self._deduplicate_timeline_events(events)

    def _deduplicate_timeline_events(self, events: list[dict]) -> list[dict]:
        """Group similar events (same date, type, description) into one."""
        if not events:
            return []

        # Sort by date first to grouping is easier
        events.sort(key=lambda x: x["date"], reverse=True)
        
        grouped_events = []
        current_event = None
        
        for event in events:
            if current_event is None:
                current_event = event
                continue
                
            # Check if this event matches the current one
            is_match = (
                event["date"] == current_event["date"] and
                event["event_type"] == current_event["event_type"] and
                # Compare descriptions loosely or just type/date
                # Often description contains the App Number, so we should strip it or compare parts
                # Let's compare the core parts of description if possible, or just ignore description if date/type match for supplements
                (
                    event["description"] == current_event["description"] or
                    ("Supplemental Application" in event["description"] and "Supplemental Application" in current_event["description"])
                )
            )
            
            if is_match:
                # Merge application numbers if both events have them
                if "application_number" in event and "application_number" in current_event:
                    if event["application_number"] not in current_event["application_number"]:
                         current_event["application_number"] += f", {event['application_number']}"
                
                # Improve description if it was generic
                if "Supplemental Application" in current_event["description"]:
                     # If grouping supplements, ensure description is clean
                     pass
            else:
                grouped_events.append(current_event)
                current_event = event
        
        if current_event:
            grouped_events.append(current_event)
            
        return grouped_events

    def _get_event_type(self, submission) -> str:
        """Determine the event type from a submission."""
        sub_type = submission.submission_type or ""
        status = submission.submission_status or ""

        if sub_type == "ORIG" and status == "AP":
            return "initial_approval"
        elif sub_type == "SUPPL" and status == "AP":
            return "supplemental_approval"
        elif status == "AP":
            return "approval"
        elif status == "TA":
            return "tentative_approval"
        else:
            return "submission"

    def _get_event_description(self, approval, submission) -> str:
        """Generate a description for a submission event."""
        parts = []

        # Application type and number
        # if approval.application_type:
        #     parts.append(f"{approval.application_type} {approval.application_number}")
        # else:
        #     parts.append(approval.application_number)
        # Note: We move App Number display to the event metadata, not description, to allow grouping.

        # Submission type
        if submission.submission_type == "ORIG":
            parts.append("Original Application")
        elif submission.submission_type == "SUPPL":
            parts.append("Supplemental Application")
            if submission.submission_class_code_description:
                parts.append(f"({submission.submission_class_code_description})")

        # Status
        status_map = {
            "AP": "Approved",
            "TA": "Tentatively Approved",
            "WD": "Withdrawn",
            "UN": "Unknown",
        }
        status = status_map.get(submission.submission_status, submission.submission_status)
        if status:
            parts.append(f"- {status}")

        # Review priority
        if submission.review_priority:
            parts.append(f"[{submission.review_priority}]")

        return " ".join(parts)

    def _deduplicate_indications(
        self,
        indications: list[DrugIndication],
    ) -> list[DrugIndication]:
        """Remove duplicate indications based on text similarity.

        Args:
            indications: List of indications to deduplicate

        Returns:
            Deduplicated list of indications
        """
        seen_texts = set()
        unique = []

        for indication in indications:
            # Normalize text for comparison
            normalized = indication.indication_text.lower().strip()[:500]

            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(indication)

        return unique

    async def get_drug_by_application(
        self,
        application_number: str,
    ) -> Optional[DrugApproval]:
        """Get drug details by FDA application number.

        Args:
            application_number: FDA application number (NDA/ANDA/BLA)

        Returns:
            DrugApproval object or None
        """
        return await self.openfda_service.get_drug_by_application_number(
            application_number
        )
