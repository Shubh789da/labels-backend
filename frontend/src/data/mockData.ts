import type { DrugHistoryResponse } from "../types";
import { DataSource } from "../types";

export const mockDrugHistory: DrugHistoryResponse = {
    drug_name: "Keytruda",
    normalized_name: "Keytruda",
    rxcui: "1547553",
    approvals: [
        {
            application_number: "BLA125514",
            sponsor_name: "MERCK SHARP DOHME",
            application_type: "BLA",
            initial_approval_date: "20140904",
            source: DataSource.OPENFDA_DRUGSFDA,
            products: [
                {
                    brand_name: "KEYTRUDA",
                    generic_name: "PEMBROLIZUMAB",
                    dosage_form: "INJECTION",
                    route: "INTRAVENOUS",
                    strength: "100MG/4ML (25MG/ML)",
                    marketing_status: "Prescription",
                    active_ingredients: ["PEMBROLIZUMAB"],
                },
            ],
            submissions: [
                {
                    submission_type: "ORIG",
                    submission_number: "000",
                    submission_status: "AP",
                    submission_status_date: "20140904",
                    submission_class_code: "UNKNOWN",
                    submission_class_code_description: "Original Application",
                    review_priority: "PRIORITY",
                    application_docs: [],
                },
                {
                    submission_type: "SUPPL",
                    submission_number: "001",
                    submission_status: "AP",
                    submission_status_date: "20151002",
                    submission_class_code_description: "New Indication",
                    application_docs: [],
                },
            ],
        },
    ],
    indications: [
        {
            indication_text:
                "KEYTRUDA is displayed for the treatment of patients with unresectable or metastatic melanoma.",
            approval_date: "20140904",
            source: DataSource.OPENFDA_LABEL,
            is_original: true,
        },
        {
            indication_text:
                "KEYTRUDA is indicated for the treatment of patients with metastatic non-small cell lung cancer (NSCLC).",
            approval_date: "20151002",
            source: DataSource.OPENFDA_LABEL,
            is_original: false,
        },
    ],
    timeline: [
        {
            date: "20140904",
            event_type: "initial_approval",
            description: "Original Application - Approved [PRIORITY]. KEYTRUDA is displayed for the treatment of patients with unresectable or metastatic melanoma.",
            application_number: "BLA125514",
            sponsor: "MERCK SHARP DOHME",
            source: "openfda_drugsfda",
        },
        {
            date: "20151002",
            event_type: "supplemental_approval",
            description: "Supplemental Application (New Indication) - Approved. KEYTRUDA is indicated for the treatment of patients with metastatic non-small cell lung cancer (NSCLC).",
            application_number: "BLA125514",
            sponsor: "MERCK SHARP DOHME",
            source: "openfda_drugsfda",
        },
        {
            date: "20160805",
            event_type: "supplemental_approval",
            description: "Supplemental Application (New Indication) - Approved. Head and Neck Squamous Cell Cancer.",
            application_number: "BLA125514",
            sponsor: "MERCK SHARP DOHME",
            source: "openfda_drugsfda",
        },
        {
            date: "20170314",
            event_type: "supplemental_approval",
            description: "Supplemental Application (New Indication) - Approved. Classical Hodgkin Lymphoma.",
            application_number: "BLA125514",
            sponsor: "MERCK SHARP DOHME",
            source: "openfda_drugsfda",
        }
    ],
    sources_queried: ["RxNorm", "openFDA Drugs@FDA", "openFDA Drug Labels"],
    errors: [],
};
