import type { DrugHistoryResponse, DrugSearchResult } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export const searchDrugs = async (term: string): Promise<DrugSearchResult[]> => {
    const response = await fetch(`${API_BASE_URL}/drugs/search?name=${encodeURIComponent(term)}`);
    if (!response.ok) {
        throw new Error('Failed to search drugs');
    }
    return response.json();
};

export const getDrugHistory = async (drugName: string): Promise<DrugHistoryResponse> => {
    // Try to clean the name or use strict search if needed, but the backend accepts drug_name
    const response = await fetch(`${API_BASE_URL}/drugs/history/${encodeURIComponent(drugName)}`);
    if (!response.ok) {
        if (response.status === 404) {
            throw new Error('Drug not found');
        }
        throw new Error('Failed to fetch drug history');
    }
    return response.json();
};

export const extractIndication = async (pdfUrl: string, filenamePrefix: string): Promise<{ found: boolean; text: string; indication_count: number }> => {
    const response = await fetch(`${API_BASE_URL}/drugs/extract-indication`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pdf_url: pdfUrl, filename_prefix: filenamePrefix }),
    });

    if (!response.ok) {
        throw new Error('Failed to extract indication');
    }
    return response.json();
};
