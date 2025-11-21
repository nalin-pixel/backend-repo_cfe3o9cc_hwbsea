import os
import base64
import time
from io import BytesIO
from typing import Optional, List, Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(default="")
    steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=50.0)
    seed: Optional[int] = Field(default=None)
    width: int = Field(default=512, ge=64, le=1536)
    height: int = Field(default=512, ge=64, le=1536)
    provider: Literal["mock", "stability", "replicate"] = Field(default="mock")


class GenerateResponse(BaseModel):
    image: str  # data URL or https URL
    provider_used: str
    latency_ms: int


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# Helper: create a simple placeholder image as SVG and return as data URL

def svg_placeholder(prompt: str, w: int, h: int, seed: Optional[int]) -> str:
    safe_prompt = (prompt[:120] + "…") if len(prompt) > 120 else prompt
    bg = "#0f172a"  # slate-900
    fg = "#60a5fa"  # blue-400
    accent = "#1e293b"  # slate-800
    ts = int(time.time())
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
      <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='{bg}'/>
          <stop offset='100%' stop-color='{accent}'/>
        </linearGradient>
      </defs>
      <rect width='100%' height='100%' fill='url(#g)'/>
      <g>
        <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' 
              font-family='Inter, system-ui, -apple-system, Segoe UI, Roboto' 
              font-size='{max(14, min(w, h)//18)}' fill='{fg}' opacity='0.9' 
              style='white-space: pre-wrap;'>
          {safe_prompt}
        </text>
      </g>
      <g>
        <text x='12' y='{h-12}' font-size='12' fill='{fg}' opacity='0.6'>seed={seed or 'auto'} • mock • {ts}</text>
      </g>
    </svg>
    """.strip()
    data = svg.encode("utf-8")
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    start = time.time()

    # Provider selection
    provider_used = "mock"
    image_url: Optional[str] = None

    if req.provider == "stability":
        api_key = os.getenv("STABILITY_API_KEY")
        if not api_key:
            provider_used = "mock"
        else:
            try:
                # SDXL Text-to-Image via Stability API v1beta (simple JSON response)
                endpoint = "https://api.stability.ai/v2beta/stable-image/generate/sd3"  # example endpoint; may vary
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }
                # Some Stability endpoints expect multipart/form-data image output (binary). To keep a simple demo
                # we will use an endpoint that returns JSON or gracefully fall back to mock if it fails.
                payload = {
                    "mode": "text-to-image",
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt or None,
                    "width": req.width,
                    "height": req.height,
                    "steps": req.steps,
                    "cfg_scale": req.guidance_scale,
                    "seed": req.seed,
                }
                r = requests.post(endpoint, json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        # Assume the API returns base64 or url in a field; this may vary per plan/version
                        if isinstance(data, dict):
                            if "image" in data:
                                image_url = data["image"]
                            elif "images" in data and data["images"]:
                                image_url = data["images"][0]
                    except Exception:
                        # Some endpoints return binary image; convert to data URL
                        b64 = base64.b64encode(r.content).decode("utf-8")
                        image_url = f"data:image/png;base64,{b64}"
                    provider_used = "stability"
                else:
                    provider_used = "mock"
            except Exception:
                provider_used = "mock"

    if req.provider == "replicate" and image_url is None:
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            provider_used = "mock"
        else:
            try:
                # Minimal Replicate run (text-to-image). Model selection may vary; using SDXL as example
                run_endpoint = "https://api.replicate.com/v1/predictions"
                headers = {
                    "Authorization": f"Token {token}",
                    "Content-Type": "application/json",
                }
                body = {
                    "version": "a16f8f8e1f4b9b7c9eaa83c8bbd3b9a2ff0d3b9a98b3f8e8f2c2e1b8e7d9f0a1",  # placeholder version id
                    "input": {
                        "prompt": req.prompt,
                        "negative_prompt": req.negative_prompt or "",
                        "width": req.width,
                        "height": req.height,
                        "num_inference_steps": req.steps,
                        "guidance_scale": req.guidance_scale,
                        "seed": req.seed,
                    },
                }
                r = requests.post(run_endpoint, json=body, headers=headers, timeout=30)
                if r.status_code in (200, 201):
                    pred = r.json()
                    # Simplified polling: if output is immediately available use it; otherwise fall back to mock
                    output = pred.get("output") if isinstance(pred, dict) else None
                    if output:
                        if isinstance(output, list) and output:
                            image_url = output[0]
                        elif isinstance(output, str):
                            image_url = output
                        provider_used = "replicate"
                    else:
                        provider_used = "mock"
                else:
                    provider_used = "mock"
            except Exception:
                provider_used = "mock"

    if image_url is None:
        image_url = svg_placeholder(req.prompt, req.width, req.height, req.seed)
        provider_used = "mock"

    latency_ms = int((time.time() - start) * 1000)
    return GenerateResponse(image=image_url, provider_used=provider_used, latency_ms=latency_ms)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
