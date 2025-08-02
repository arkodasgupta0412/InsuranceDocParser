from fastapi import FastAPI
from routes import base_url

app = FastAPI(title="Insurance Query Solver", version="1.0.0")

app.include_router(router = base_url.router)
