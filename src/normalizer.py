from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Review:
    texto: str
    rating: Optional[float] = None      # 1-5, None si no viene
    plataforma: Optional[str] = None    # "google" | "booking" | "rest-mex" | "csv"
    fecha: Optional[datetime] = None
    id_externo: Optional[str] = None   


def from_restmex(row: dict) -> Review:
   #Convierte una fila del dataset Rest-Mex al esquema común.
    return Review(
        texto=row.get("review", row.get("text", "")),
        rating=row.get("rating", None),
        plataforma="rest-mex",
        fecha=None,
        id_externo=str(row.get("id", "")),
    )


def from_google(place: dict) -> Review:
   #Convierte respuesta de Google Places API al esquema común.
    return Review(
        texto=place.get("text", {}).get("text", ""),
        rating=place.get("rating", None),
        plataforma="google",
        fecha=None,
        id_externo=place.get("name", ""),
    )


def from_csv(row: dict) -> Review:
    #Convierte fila de CSV manual al esquema común.
    return Review(
        texto=row.get("texto", ""),
        rating=float(row["rating"]) if "rating" in row and row["rating"] else None,
        plataforma=row.get("plataforma", "csv"),
        fecha=None,
    )