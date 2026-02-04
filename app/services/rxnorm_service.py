"""Service for interacting with the RxNorm API."""
import logging
from typing import Optional

from app.services.base_service import BaseAPIService
from app.models import DrugSearchResult

logger = logging.getLogger(__name__)


class RxNormService(BaseAPIService):
    """Service for querying the RxNorm API for drug identification and normalization.

    Documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html

    RxNorm provides normalized names for clinical drugs and links to many drug vocabularies.
    Rate limit: 20 requests per second per IP address.
    """

    @property
    def base_url(self) -> str:
        return self.settings.RXNORM_BASE_URL

    async def search_drugs(self, drug_name: str) -> list[DrugSearchResult]:
        """Search for drugs by name and get RxCUI identifiers.

        Uses the getDrugs endpoint which searches across ingredient names,
        brand names, and dose forms.

        Args:
            drug_name: Drug name to search for

        Returns:
            List of matching drug results with RxCUI identifiers
        """
        data = await self._get("/drugs.json", params={"name": drug_name})

        if not data:
            logger.info(f"No RxNorm results for: {drug_name}")
            return []

        results = []
        drug_group = data.get("drugGroup", {})

        # Get concept group - contains different term types
        concept_groups = drug_group.get("conceptGroup", [])

        for group in concept_groups:
            concept_properties = group.get("conceptProperties", [])
            for prop in concept_properties:
                results.append(
                    DrugSearchResult(
                        rxcui=prop.get("rxcui"),
                        name=prop.get("name", ""),
                        synonym=prop.get("synonym"),
                        tty=prop.get("tty"),  # Term type
                    )
                )

        return results

    async def get_rxcui_by_name(self, drug_name: str) -> Optional[str]:
        """Get the RxCUI for a drug by exact name match.

        Args:
            drug_name: Drug name to look up

        Returns:
            RxCUI string or None if not found
        """
        data = await self._get("/rxcui.json", params={"name": drug_name})

        if not data:
            return None

        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId", [])

        return rxnorm_ids[0] if rxnorm_ids else None

    async def get_approximate_match(
        self,
        drug_name: str,
        max_entries: int = 5,
    ) -> list[DrugSearchResult]:
        """Get approximate matches for a drug name.

        Useful when exact name matching fails. Uses approximate term matching.

        Args:
            drug_name: Drug name to search for
            max_entries: Maximum number of results

        Returns:
            List of approximate matches
        """
        params = {
            "term": drug_name,
            "maxEntries": max_entries,
        }

        data = await self._get("/approximateTerm.json", params=params)

        if not data:
            return []

        results = []
        approx_group = data.get("approximateGroup", {})
        candidates = approx_group.get("candidate", [])

        for candidate in candidates:
            results.append(
                DrugSearchResult(
                    rxcui=candidate.get("rxcui"),
                    name=candidate.get("name", ""),
                    synonym=None,
                    tty=candidate.get("tty"),
                )
            )

        return results

    async def get_drug_properties(self, rxcui: str) -> Optional[dict]:
        """Get properties for a drug by RxCUI.

        Args:
            rxcui: RxNorm Concept Unique Identifier

        Returns:
            Dictionary of drug properties or None
        """
        data = await self._get(f"/rxcui/{rxcui}/properties.json")

        if not data:
            return None

        return data.get("properties")

    async def get_related_drugs(
        self,
        rxcui: str,
        relation_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """Get drugs related to a given RxCUI.

        Args:
            rxcui: RxNorm Concept Unique Identifier
            relation_types: Types of relations to include (e.g., ["has_ingredient"])

        Returns:
            List of related drug concepts
        """
        endpoint = f"/rxcui/{rxcui}/related.json"
        params = {}

        if relation_types:
            params["tty"] = "+".join(relation_types)

        data = await self._get(endpoint, params=params if params else None)

        if not data:
            return []

        related = []
        related_group = data.get("relatedGroup", {})
        concept_groups = related_group.get("conceptGroup", [])

        for group in concept_groups:
            concept_properties = group.get("conceptProperties", [])
            for prop in concept_properties:
                related.append(
                    {
                        "rxcui": prop.get("rxcui"),
                        "name": prop.get("name"),
                        "tty": prop.get("tty"),
                        "language": prop.get("language"),
                    }
                )

        return related

    async def get_ndcs_for_drug(self, rxcui: str) -> list[str]:
        """Get National Drug Codes (NDCs) for a drug by RxCUI.

        Args:
            rxcui: RxNorm Concept Unique Identifier

        Returns:
            List of NDC codes
        """
        data = await self._get(f"/rxcui/{rxcui}/ndcs.json")

        if not data:
            return []

        ndc_group = data.get("ndcGroup", {})
        return ndc_group.get("ndcList", {}).get("ndc", [])

    async def normalize_drug_name(self, drug_name: str) -> Optional[dict]:
        """Normalize a drug name using RxNorm.

        Attempts to find the best matching normalized drug name.

        Args:
            drug_name: Drug name to normalize

        Returns:
            Dictionary with normalized name and RxCUI, or None
        """
        # Try exact match first
        rxcui = await self.get_rxcui_by_name(drug_name)

        if rxcui:
            properties = await self.get_drug_properties(rxcui)
            if properties:
                return {
                    "rxcui": rxcui,
                    "normalized_name": properties.get("name"),
                    "term_type": properties.get("tty"),
                    "match_type": "exact",
                }

        # Try getDrugs endpoint
        drugs = await self.search_drugs(drug_name)
        if drugs:
            # Prefer ingredient (IN) or brand name (BN) term types
            preferred_ttys = ["IN", "BN", "SBD", "SCD"]
            for tty in preferred_ttys:
                for drug in drugs:
                    if drug.tty == tty:
                        return {
                            "rxcui": drug.rxcui,
                            "normalized_name": drug.name,
                            "term_type": drug.tty,
                            "match_type": "search",
                        }
            # Return first result if no preferred type found
            return {
                "rxcui": drugs[0].rxcui,
                "normalized_name": drugs[0].name,
                "term_type": drugs[0].tty,
                "match_type": "search",
            }

        # Try approximate matching as last resort
        approx_matches = await self.get_approximate_match(drug_name)
        if approx_matches:
            return {
                "rxcui": approx_matches[0].rxcui,
                "normalized_name": approx_matches[0].name,
                "term_type": approx_matches[0].tty,
                "match_type": "approximate",
            }

        return None
