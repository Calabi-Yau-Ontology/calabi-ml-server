from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.nlp.router import router as nlp_router
from src.ontology.router import router as ontology_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        debug=settings.DEBUG,
        version=settings.VERSION,
        port=settings.PORT,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(nlp_router)
    app.include_router(ontology_router)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app

app = create_app()
