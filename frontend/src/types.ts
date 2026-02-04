export type DataSource =
    | "openfda_drugsfda"
    | "openfda_label"
    | "dailymed"
    | "rxnorm";

export const DataSource = {
    OPENFDA_DRUGSFDA: "openfda_drugsfda" as DataSource,
    OPENFDA_LABEL: "openfda_label" as DataSource,
    DAILYMED: "dailymed" as DataSource,
    RXNORM: "rxnorm" as DataSource,
};

export interface ApplicationDocument {
    id?: string;
    url?: string;
    date?: string;
    type?: string;
}

export interface DrugSubmission {
    submission_type?: string;
    submission_number?: string;
    submission_status?: string;
    submission_status_date?: string;
    submission_class_code?: string;
    submission_class_code_description?: string;
    review_priority?: string;
    application_docs: ApplicationDocument[];
}

export interface DrugIndication {
    indication_text: string;
    approval_date?: string;
    source: DataSource;
    url?: string;
    is_original: boolean;
}

export interface DrugProduct {
    brand_name?: string;
    generic_name?: string;
    dosage_form?: string;
    route?: string;
    strength?: string;
    marketing_status?: string;
    active_ingredients: string[];
}

export interface DrugApproval {
    application_number: string;
    sponsor_name?: string;
    application_type?: string;
    submissions: DrugSubmission[];
    products: DrugProduct[];
    initial_approval_date?: string;
    source: DataSource;
}

export interface TimelineEvent {
    date: string;
    event_type: string;
    description: string;
    application_number?: string;
    sponsor?: string;
    source: string;
    url?: string;
    is_loading?: boolean;
    filename_prefix?: string;
}

export interface KeyDocument {
    title: string;
    date: string;
    url: string;
    type: string;
}

export interface DrugHistoryResponse {
    drug_name: string;
    normalized_name?: string;
    rxcui?: string;
    approvals: DrugApproval[];
    indications: DrugIndication[];
    indication_count?: number;
    key_documents?: KeyDocument[];
    timeline: TimelineEvent[];
    sources_queried: string[];
    errors: string[];
}

export interface DrugSearchResult {
    rxcui: string;
    name: string;
    synonym?: string;
    tty?: string;
}
