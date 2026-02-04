# Drug Approval History API

A FastAPI backend that aggregates drug approval history from multiple public data sources including FDA, DailyMed, and RxNorm.

## Features

- **Drug Search**: Search for drugs by name with automatic normalization via RxNorm
- **Approval History**: Get FDA approval records including all submissions and dates
- **Indications**: View therapeutic indications from FDA drug labeling
- **Timeline**: Chronological view of all approval events
- **Application Lookup**: Search by FDA application number (NDA/ANDA/BLA)

## Data Sources

| Source | Description | Documentation |
|--------|-------------|---------------|
| [openFDA Drugs@FDA](https://open.fda.gov/apis/drug/drugsfda/) | FDA drug approval records since 1939 | [API Docs](https://open.fda.gov/apis/) |
| [openFDA Drug Labels](https://open.fda.gov/apis/drug/label/) | Structured product labeling with indications | [API Docs](https://open.fda.gov/apis/) |
| [DailyMed](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) | NLM's FDA drug labeling database | [Web Services](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) |
| [RxNorm](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html) | NLM standardized drug naming system | [API Docs](https://lhncbc.nlm.nih.gov/RxNav/APIs/index.html) |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Search Drugs
```
GET /drugs/search?name={drug_name}
```
Search for drugs using RxNorm to get standardized identifiers.

### Get Drug History
```
GET /drugs/history/{drug_name}?include_dailymed=true
```
Get comprehensive drug approval history from all sources.

**Example Response:**
```json
{
  "drug_name": "metformin",
  "normalized_name": "metformin",
  "rxcui": "6809",
  "approvals": [...],
  "indications": [...],
  "timeline": [...],
  "sources_queried": ["RxNorm", "openFDA Drugs@FDA", "openFDA Drug Labels", "DailyMed"]
}
```

### Get Drug Approvals
```
GET /drugs/approvals?drug_name={drug_name}&limit=20
```
Search FDA drug approvals database directly.

### Get Drug Indications
```
GET /drugs/indications/{drug_name}
```
Get therapeutic indications from FDA labeling.

### Lookup by Application Number
```
GET /drugs/application/{application_number}
```
Get drug details by FDA application number (e.g., NDA012345).

### Normalize Drug Name
```
GET /drugs/normalize/{drug_name}
```
Get standardized drug name from RxNorm.

## Configuration

Create a `.env` file for optional configuration:

```env
# Optional: openFDA API key for higher rate limits
OPENFDA_API_KEY=your_api_key_here

# Cache TTL in seconds (default: 3600)
CACHE_TTL=3600

# Request timeout in seconds (default: 30)
REQUEST_TIMEOUT=30

# Debug mode (default: false)
DEBUG=false
```

## Rate Limits

The API relies on external services with their own rate limits:

| Service | Rate Limit |
|---------|------------|
| openFDA (no key) | 240 requests/minute |
| openFDA (with key) | 120,000 requests/day |
| RxNorm | 20 requests/second |
| DailyMed | No documented limit |

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Project Structure

```
drug_history/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── README.md
└── app/
    ├── __init__.py
    ├── config.py           # Configuration settings
    ├── models/
    │   ├── __init__.py
    │   └── drug.py         # Pydantic data models
    ├── routes/
    │   ├── __init__.py
    │   └── drugs.py        # API route handlers
    └── services/
        ├── __init__.py
        ├── base_service.py      # Base HTTP service class
        ├── openfda_service.py   # openFDA API integration
        ├── dailymed_service.py  # DailyMed API integration
        ├── rxnorm_service.py    # RxNorm API integration
        └── aggregator_service.py # Data aggregation service
```

## Example Usage

### Using curl

```bash
# Search for a drug
curl "http://localhost:8000/drugs/search?name=aspirin"

# Get complete approval history
curl "http://localhost:8000/drugs/history/metformin"

# Get indications only
curl "http://localhost:8000/drugs/indications/lisinopril"

# Lookup by NDA number
curl "http://localhost:8000/drugs/application/NDA020357"
```

### Using Python

```python
import httpx

async with httpx.AsyncClient() as client:
    # Get drug history
    response = await client.get(
        "http://localhost:8000/drugs/history/metformin"
    )
    data = response.json()

    print(f"Drug: {data['normalized_name']}")
    print(f"RxCUI: {data['rxcui']}")
    print(f"Approvals: {len(data['approvals'])}")

    # Print timeline
    for event in data['timeline'][:5]:
        print(f"  {event['date']}: {event['event_type']}")
```

## License

This project uses public government data sources. Please review the terms of use for each data source:
- [openFDA Terms](https://open.fda.gov/license/)
- [NLM Terms](https://www.nlm.nih.gov/databases/download.html)
