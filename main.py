"""Drug Approval History API - FastAPI Application.

This API provides access to drug approval history data from multiple public sources:
- openFDA (Drugs@FDA and Drug Labels)
- DailyMed (NLM)
- RxNorm (NLM)

Run with: uvicorn main:app --reload
For debug logging: uvicorn main:app --reload --log-level debug
"""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.routes import drugs_router

# Configure logging - DEBUG level shows RunPod API URLs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## Drug Approval History API

This API aggregates drug approval information from multiple public data sources
to provide comprehensive approval history, indications, and timeline data.

### Data Sources

- **openFDA Drugs@FDA**: FDA drug approval records since 1939
- **openFDA Drug Labels**: Structured product labeling with indication information
- **DailyMed**: NLM's database of FDA drug labeling
- **RxNorm**: NLM's standardized drug naming system

### Key Features

- Search drugs by name with automatic name normalization
- Get complete approval history including all submissions
- View therapeutic indications from FDA labeling
- Chronological timeline of approval events
- Lookup by FDA application number (NDA/ANDA/BLA)

### Usage Notes

- Drug names are automatically normalized using RxNorm for better matching
- Results are aggregated from multiple sources and deduplicated
- Approval dates come from FDA submission records
- Indication text is extracted from drug labeling

### Rate Limits

This API relies on public APIs with their own rate limits:
- openFDA: No key = 240 requests/minute, With key = 120,000 requests/day
- RxNorm: 20 requests/second per IP
- DailyMed: No documented limits, please use responsibly
    """,
    openapi_tags=[
        {
            "name": "drugs",
            "description": "Drug approval history and indication endpoints",
        },
    ],
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(drugs_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Drug approval history API aggregating data from openFDA, DailyMed, and RxNorm",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "search_drugs": "/drugs/search?name={drug_name}",
            "drug_history": "/drugs/history/{drug_name}",
            "drug_approvals": "/drugs/approvals?drug_name={drug_name}",
            "drug_indications": "/drugs/indications/{drug_name}",
            "normalize_name": "/drugs/normalize/{drug_name}",
            "by_application": "/drugs/application/{application_number}",
        },
        "data_sources": [
            {
                "name": "openFDA",
                "url": "https://open.fda.gov/",
                "description": "FDA drug approval data and labeling",
            },
            {
                "name": "DailyMed",
                "url": "https://dailymed.nlm.nih.gov/",
                "description": "NLM drug labeling database",
            },
            {
                "name": "RxNorm",
                "url": "https://www.nlm.nih.gov/research/umls/rxnorm/",
                "description": "NLM standardized drug naming",
            },
        ],
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.APP_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
