import logging
import sys
from pathlib import Path
from typing import Optional

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.database import get_connection


# The final inference modules are intentionally kept as the single source of
# truth under src/. This project-relative path avoids machine-specific paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_review import analyze_review as run_review_analysis  # noqa: E402


logger = logging.getLogger(__name__)
app = FastAPI(title="Neden Gidelim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ScoresInput(BaseModel):
    taste: float = Field(ge=0, le=10)
    atmosphere: float = Field(ge=0, le=10)
    service: float = Field(ge=0, le=10)
    staff: float = Field(ge=0, le=10)
    value: float = Field(ge=0, le=10)
    decoration: float = Field(ge=0, le=10)


class PlaceInput(BaseModel):
    name: str
    district: str
    neighborhood: str
    category: str
    overallScore: float = Field(ge=0, le=10)
    priceLevel: str
    coffeeBeans: Optional[str] = None
    aiSummary: str
    scores: ScoresInput


class ReviewAnalysisRequest(BaseModel):
    text: str


SELECT_PLACES_QUERY = """
    SELECT
        id,
        name,
        district,
        neighborhood,
        category,
        overall_score,
        price_level,
        coffee_beans,
        ai_summary,
        taste_score,
        atmosphere_score,
        service_score,
        staff_score,
        value_score,
        decoration_score
    FROM places
    ORDER BY id;
"""


SELECT_PLACE_BY_ID_QUERY = """
    SELECT
        id,
        name,
        district,
        neighborhood,
        category,
        overall_score,
        price_level,
        coffee_beans,
        ai_summary,
        taste_score,
        atmosphere_score,
        service_score,
        staff_score,
        value_score,
        decoration_score
    FROM places
    WHERE id = %s;
"""


INSERT_PLACE_QUERY = """
    INSERT INTO places (
        name,
        district,
        neighborhood,
        category,
        overall_score,
        price_level,
        coffee_beans,
        ai_summary,
        taste_score,
        atmosphere_score,
        service_score,
        staff_score,
        value_score,
        decoration_score
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s
    )
    RETURNING
        id,
        name,
        district,
        neighborhood,
        category,
        overall_score,
        price_level,
        coffee_beans,
        ai_summary,
        taste_score,
        atmosphere_score,
        service_score,
        staff_score,
        value_score,
        decoration_score;
"""


def format_place(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "district": row["district"],
        "neighborhood": row["neighborhood"],
        "category": row["category"],
        "scores": {
            "taste": float(row["taste_score"]),
            "atmosphere": float(row["atmosphere_score"]),
            "service": float(row["service_score"]),
            "staff": float(row["staff_score"]),
            "value": float(row["value_score"]),
            "decoration": float(row["decoration_score"]),
        },
        "overallScore": float(row["overall_score"]),
        "priceLevel": row["price_level"],
        "coffeeBeans": row["coffee_beans"],
        "aiSummary": row["ai_summary"],
    }


@app.get("/places")
def get_places():
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(SELECT_PLACES_QUERY)
                rows = cursor.fetchall()
    except (psycopg.Error, RuntimeError) as error:
        print("Veritabanı hatası:", error)
        raise HTTPException(
            status_code=503,
            detail="Veritabanına bağlanılamadı.",
        ) from error

    return [format_place(row) for row in rows]


@app.get("/places/{place_id}")
def get_place(place_id: int):
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(SELECT_PLACE_BY_ID_QUERY, (place_id,))
                row = cursor.fetchone()
    except (psycopg.Error, RuntimeError) as error:
        print("Veritabanı hatası:", error)
        raise HTTPException(
            status_code=503,
            detail="Veritabanına bağlanılamadı.",
        ) from error

    if row is None:
        raise HTTPException(status_code=404, detail="Mekan bulunamadı.")

    return format_place(row)


@app.post("/places", status_code=201)
def create_place(place: PlaceInput):
    values = (
        place.name,
        place.district,
        place.neighborhood,
        place.category,
        place.overallScore,
        place.priceLevel,
        place.coffeeBeans,
        place.aiSummary,
        place.scores.taste,
        place.scores.atmosphere,
        place.scores.service,
        place.scores.staff,
        place.scores.value,
        place.scores.decoration,
    )

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(INSERT_PLACE_QUERY, values)
                row = cursor.fetchone()
    except (psycopg.Error, RuntimeError) as error:
        print("Veritabanı hatası:", error)
        raise HTTPException(
            status_code=503,
            detail="Veritabanına bağlanılamadı.",
        ) from error

    return format_place(row)


@app.post("/analyze-review")
def analyze_review_endpoint(review: ReviewAnalysisRequest):
    text = review.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="text must be a non-empty string",
        )

    try:
        return run_review_analysis(text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Review analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Review analysis failed.",
        ) from error
