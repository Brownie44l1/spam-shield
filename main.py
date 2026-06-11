from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router
import os

app = FastAPI(
    title="SpamShield API",
    description="ML-powered spam email detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
BASE_DIR = os.path.dirname(__file__)
static_dir = os.path.join(BASE_DIR, "app/static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app/templates"))

# Routes
app.include_router(predict_router, prefix="/api")


@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.head("/ping")
def ping_head():
    return {}

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
