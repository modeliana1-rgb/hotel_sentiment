# Hotel Sentiment — Sistema de análisis de reputación hotelera

## Setup
```bash
conda env create -f environment.yml
conda activate hotel_sentiment
```

## Estructura
| Carpeta | Responsable | Descripción |
|---|---|---|
| `src/` | Persona 1 | Normalizer + carga de datos |
| `models/` | Persona 2 | 4 clasificadores + evaluación |
| `pipeline/` | Persona 3 | Orchestrator + ABSA + API |
| `app/` | Persona 4 | Dashboard Streamlit |
| `notebooks/` | Persona 1 | EDA + comparación modelos |

## Arrancar la API
```bash
uvicorn pipeline.api:app --reload
```

## Arrancar el dashboard
```bash
streamlit run app/dashboard.py
```