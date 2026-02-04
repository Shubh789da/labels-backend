"""Service for interacting with the DailyMed API."""
import logging
from typing import Optional

from app.services.base_service import BaseAPIService
from app.models import DrugIndication, DataSource

logger = logging.getLogger(__name__)


class DailyMedService(BaseAPIService):
    """Service for querying the DailyMed API for drug labeling information.

    Documentation: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

    DailyMed provides access to FDA-approved drug labeling (Structured Product Labeling/SPL).
    """

    @property
    def base_url(self) -> str:
        return self.settings.DAILYMED_BASE_URL

    async def search_drug_spls(
        self,
        drug_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """Search for SPL (Structured Product Labeling) records by drug name.

        Args:
            drug_name: Drug name to search for
            limit: Maximum number of results

        Returns:
            List of SPL records with basic info
        """
        params = {
            "drug_name": drug_name,
            "pagesize": min(limit, 100),  # API max is 100
        }

        data = await self._get("/spls.json", params=params)

        if not data or "data" not in data:
            logger.info(f"No DailyMed SPL results for: {drug_name}")
            return []

        return data.get("data", [])

    async def get_spl_by_setid(self, setid: str) -> Optional[dict]:
        """Get detailed SPL information by set ID.
        
        Args:
            setid: The unique SPL set identifier
            
        Returns:
            SPL details or None if not found
        """
        try:
            # DailyMed requires specific headers for JSON endpoints
            headers = {"Accept": "application/json"}
            data = await self._get(f"/spls/{setid}.json", headers=headers)
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch SPL details for {setid}: {e}")
            return None

    async def search_by_application_number(
        self,
        application_number: str,
        limit: int = 20,
    ) -> list[dict]:
        """Search SPLs by FDA application number (NDA/ANDA/BLA).

        Args:
            application_number: FDA application number
            limit: Maximum number of results

        Returns:
            List of matching SPL records
        """
        # Remove any prefix formatting inconsistencies
        clean_number = application_number.upper().strip()

        params = {
            "application_number": clean_number,
            "pagesize": min(limit, 100),
        }

        data = await self._get("/spls.json", params=params)

        if not data or "data" not in data:
            return []

        return data.get("data", [])

    async def get_drug_indications(
        self,
        drug_name: str,
        limit: int = 10,
    ) -> list[DrugIndication]:
        """Get indication information for a drug from DailyMed.

        This searches SPLs and extracts indication-related information
        from the labeling data.

        Args:
            drug_name: Drug name to search for
            limit: Maximum number of SPLs to process

        Returns:
            List of DrugIndication objects
        """
        spls = await self.search_drug_spls(drug_name, limit=limit)

        if not spls:
            return []

        indications = []
        seen_texts = set()

        for spl in spls:
            # The SPL list contains basic info; title often includes indication hints
            title = spl.get("title", "")
            published_date = spl.get("published_date")

            # Extract meaningful indication info from title
            # Format is usually: "DRUG NAME- ingredient FORM, ROUTE"
            if title and title not in seen_texts:
                seen_texts.add(title)

                # Try to get more details from the SPL
                setid = spl.get("setid")
                if setid:
                    spl_details = await self.get_spl_by_setid(setid)
                    if spl_details:
                        # Look for indication sections in the detailed SPL
                        indication_text = self._extract_indication_from_spl(
                            spl_details, title
                        )
                        if indication_text and indication_text not in seen_texts:
                            seen_texts.add(indication_text)
                            indications.append(
                                DrugIndication(
                                    indication_text=indication_text,
                                    approval_date=published_date,
                                    source=DataSource.DAILYMED,
                                    is_original=len(indications) == 0,
                                )
                            )

        return indications

    def _extract_indication_from_spl(
        self,
        spl_details: dict,
        fallback_title: str,
    ) -> str:
        """Extract indication information from SPL details.

        Args:
            spl_details: Full SPL document details
            fallback_title: Title to use if no indication found

        Returns:
            Indication text
        """
        # SPL structure varies; try common fields
        # The actual indication text is usually in the full SPL XML
        # For the JSON API, we get metadata

        # Use title as a reasonable fallback for now
        # Full indication text requires parsing SPL XML
        return fallback_title

    async def get_drug_classes(
        self,
        drug_name: str,
    ) -> list[dict]:
        """Get pharmacologic class information for a drug.

        Args:
            drug_name: Drug name to search for

        Returns:
            List of drug class information
        """
        # First get SPLs for the drug
        spls = await self.search_drug_spls(drug_name, limit=5)

        if not spls:
            return []

        # DailyMed includes pharmacologic class in SPL data
        classes = []
        seen_classes = set()

        for spl in spls:
            # Drug classes may be in the SPL details
            setid = spl.get("setid")
            if setid:
                spl_details = await self.get_spl_by_setid(setid)
                if spl_details and "data" in spl_details:
                    for item in spl_details.get("data", []):
                        drug_class = item.get("drug_class")
                        if drug_class and drug_class not in seen_classes:
                            seen_classes.add(drug_class)
                            classes.append({"drug_class": drug_class})

        return classes
