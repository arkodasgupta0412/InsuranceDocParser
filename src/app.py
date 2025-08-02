from fastapi import FastAPI
from routes import base_url

app = FastAPI(title="Insurance Query Solver", version="1.0.0")


@app.get("/api/v1/test")
def get():
    return {"message": "Insurance Parser API is live at /api/v1/test"}


app.include_router(router = base_url.router)
