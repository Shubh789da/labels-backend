"""Service for interacting with the openFDA API."""
import logging
from typing import Optional
import re

from app.services.base_service import BaseAPIService
from app.models import (
    ApplicationDocument,
    DrugApproval,
    DrugSubmission,
    DrugProduct,
    DrugIndication,
    DataSource,
)

logger = logging.getLogger(__name__)


class OpenFDAService(BaseAPIService):
    """Service for querying openFDA Drugs@FDA and Drug Label APIs.

    Documentation:
    - Drugs@FDA: https://open.fda.gov/apis/drug/drugsfda/
    - Drug Label: https://open.fda.gov/apis/drug/label/
    """

    @property
    def base_url(self) -> str:
        return self.settings.OPENFDA_BASE_URL

    def _build_search_query(self, drug_name: str) -> str:
        """Build a search query for drug name across multiple fields."""
        # Escape special characters for Elasticsearch query
        escaped_name = drug_name.replace('"', '\\"')
        # Search across brand name, generic name, and substance name
        return (
            f'openfda.brand_name:"{escaped_name}" OR '
            f'openfda.generic_name:"{escaped_name}" OR '
            f'openfda.substance_name:"{escaped_name}"'
        )

    async def search_drug_approvals(
        self,
        drug_name: str,
        limit: int = 100,
    ) -> list[DrugApproval]:
        """Search for drug approval records in Drugs@FDA.

        Args:
            drug_name: Name of the drug to search for
            limit: Maximum number of results to return

        Returns:
            List of DrugApproval objects
        """
        params = {
            "search": self._build_search_query(drug_name),
            "limit": limit,
        }

        # Add API key if configured
        if self.settings.OPENFDA_API_KEY:
            params["api_key"] = self.settings.OPENFDA_API_KEY

        data = await self._get("/drug/drugsfda.json", params=params)

        if not data or "results" not in data:
            logger.info(f"No Drugs@FDA results for: {drug_name}")
            return []

        approvals = []
        for result in data["results"]:
            approval = self._parse_approval(result)
            if approval:
                approvals.append(approval)

        return approvals

    def _parse_approval(self, result: dict) -> Optional[DrugApproval]:
        """Parse a single approval result from the API."""
        try:
            # Parse submissions
            submissions = []
            for sub in result.get("submissions", []):
                # Parse application documents for this submission
                application_docs = []
                for doc in sub.get("application_docs", []):
                    application_docs.append(
                        ApplicationDocument(
                            id=doc.get("id"),
                            url=doc.get("url"),
                            date=doc.get("date"),
                            type=doc.get("type"),
                        )
                    )

                submission = DrugSubmission(
                    submission_type=sub.get("submission_type"),
                    submission_number=sub.get("submission_number"),
                    submission_status=sub.get("submission_status"),
                    submission_status_date=sub.get("submission_status_date"),
                    submission_class_code=sub.get("submission_class_code"),
                    submission_class_code_description=sub.get(
                        "submission_class_code_description"
                    ),
                    review_priority=sub.get("review_priority"),
                    application_docs=application_docs,
                )
                submissions.append(submission)

            # Parse products
            products = []
            openfda = result.get("openfda", {})
            for prod in result.get("products", []):
                # Get active ingredients
                active_ingredients = []
                for ing in prod.get("active_ingredients", []):
                    name = ing.get("name", "")
                    strength = ing.get("strength", "")
                    if name:
                        active_ingredients.append(f"{name} {strength}".strip())

                product = DrugProduct(
                    brand_name=openfda.get("brand_name", [None])[0]
                    if openfda.get("brand_name")
                    else prod.get("brand_name"),
                    generic_name=openfda.get("generic_name", [None])[0]
                    if openfda.get("generic_name")
                    else None,
                    dosage_form=prod.get("dosage_form"),
                    route=prod.get("route"),
                    strength=prod.get("te_code"),
                    marketing_status=prod.get("marketing_status"),
                    active_ingredients=active_ingredients,
                )
                products.append(product)

            # Find initial approval date (earliest ORIG submission with AP status)
            initial_approval_date = None
            orig_submissions = [
                s for s in submissions
                if s.submission_type == "ORIG" and s.submission_status == "AP"
            ]
            if orig_submissions:
                dates = [s.submission_status_date for s in orig_submissions if s.submission_status_date]
                if dates:
                    initial_approval_date = min(dates)

            # Determine application type from application number
            app_number = result.get("application_number", "")
            app_type = None
            if app_number.startswith("NDA"):
                app_type = "NDA"
            elif app_number.startswith("ANDA"):
                app_type = "ANDA"
            elif app_number.startswith("BLA"):
                app_type = "BLA"

            return DrugApproval(
                application_number=app_number,
                sponsor_name=result.get("sponsor_name"),
                application_type=app_type,
                submissions=submissions,
                products=products,
                initial_approval_date=initial_approval_date,
                source=DataSource.OPENFDA_DRUGSFDA,
            )
        except Exception as e:
            logger.error(f"Error parsing approval result: {e}")
            return None

    async def search_drug_labels(
        self,
        drug_name: str,
        limit: int = 10,
    ) -> list[DrugIndication]:
        """Search drug labels for indication information.

        Args:
            drug_name: Name of the drug to search for
            limit: Maximum number of results to return

        Returns:
            List of DrugIndication objects extracted from labels
        """
        params = {
            "search": self._build_search_query(drug_name),
            "limit": limit,
        }

        if self.settings.OPENFDA_API_KEY:
            params["api_key"] = self.settings.OPENFDA_API_KEY

        data = await self._get("/drug/label.json", params=params)

        if not data or "results" not in data:
            logger.info(f"No drug label results for: {drug_name}")
            return []

        indications = []
        seen_texts = set()

        for result in data["results"]:
            # Extract indications_and_usage field
            indications_text = result.get("indications_and_usage", [])
            if isinstance(indications_text, list):
                for text in indications_text:
                    # Clean and deduplicate
                    cleaned = self._clean_indication_text(text)
                    if cleaned and cleaned not in seen_texts:
                        seen_texts.add(cleaned)
                        
                        # Construct URL if SPL Set ID is present
                        url = None
                        openfda = result.get("openfda", {})
                        spl_set_ids = openfda.get("spl_set_id", [])
                        if spl_set_ids:
                            # Use the first SPL Set ID
                            url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={spl_set_ids[0]}"

                        indications.append(
                            DrugIndication(
                                indication_text=cleaned,
                                approval_date=result.get("effective_time"),
                                source=DataSource.OPENFDA_LABEL,
                                url=url,
                                is_original=len(indications) == 0,
                            )
                        )

        return indications

    def _clean_indication_text(self, text: str) -> str:
        """Clean and normalize indication text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate very long texts
        if len(text) > 2000:
            text = text[:2000] + "..."

        return text

    async def get_drug_by_application_number(
        self,
        application_number: str,
    ) -> Optional[DrugApproval]:
        """Get drug approval details by application number.

        Args:
            application_number: FDA application number (e.g., NDA012345)

        Returns:
            DrugApproval object or None if not found
        """
        params = {
            "search": f'application_number:"{application_number}"',
            "limit": 1,
        }

        if self.settings.OPENFDA_API_KEY:
            params["api_key"] = self.settings.OPENFDA_API_KEY

        data = await self._get("/drug/drugsfda.json", params=params)

        if not data or "results" not in data or not data["results"]:
            return None

        return self._parse_approval(data["results"][0])
