import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
APP_TITLE = "Rome Food Tours API"
TOURS_FILE = Path("./saved_tours.json")
SESSIONS: dict = {}  # in-memory chat sessions

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Netlify/GH Pages URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_tours() -> list:
    if TOURS_FILE.exists():
        return json.loads(TOURS_FILE.read_text())
    return []

def save_tours(tours: list) -> None:
    TOURS_FILE.write_text(json.dumps(tours, ensure_ascii=False, indent=2))

# ── Schemas ───────────────────────────────────────────────────────────────────
class TourPreferences(BaseModel):
    area: str = "Trastevere"
    duration_minutes: int = 180
    budget: str = "medium"          # low | medium | high
    dietary: List[str] = []
    group_type: str = "solo"
    vibe: List[str] = []
    walking_level: str = "moderate" # easy | moderate | high
    time_of_day: str = "sera"
    language: str = "en"
    constraints_note: str = ""

class PlanTourRequest(BaseModel):
    user_id: str
    city: str = "Rome"
    tour_type: str = "food"
    preferences: TourPreferences

class ChatRequest(BaseModel):
    user_id: str
    tour_id: Optional[str] = None
    session_id: Optional[str] = None
    user_message: str
    context: dict = {}

# ── Prompts ───────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are an expert local food tour guide for Rome, Italy.
Generate a structured JSON food tour itinerary based on the user's preferences.
Respond ONLY with valid JSON matching this exact schema:
{
  "tour_id": "<uuid string>",
  "title": "<evocative short title>",
  "description": "<2-3 sentence description>",
  "summary": {
    "title": "<same as above>",
    "description": "<same as above>",
    "duration_minutes": <integer>,
    "distance_km": <float>,
    "budget": "<low|medium|high>"
  },
  "stops": [
    {
      "order": 1,
      "name": "<place name>",
      "type": "<bar|trattoria|gelateria|mercato|pasticceria|etc>",
      "address": "<street address, Rome>",
      "description": "<what to eat and why this spot>",
      "tip": "<insider tip>",
      "duration_minutes": <integer>
    }
  ],
  "assistant_context": {
    "session_id": "<uuid string>"
  }
}
Include 4-6 stops. Be specific with real Roman places."""

GUIDE_SYSTEM = """You are a friendly, knowledgeable live food tour guide for Rome.
You are mid-tour with the user. Answer questions about food, culture, history,
directions, or alternatives. Be concise (2-4 sentences), warm, and helpful.
If you have tour context, reference it. Respond in the user's language."""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "app": APP_TITLE}

@app.post("/api/plan-tour")
def plan_tour(req: PlanTourRequest):
    prefs = req.preferences
    dietary_str = ", ".join(prefs.dietary) if prefs.dietary else "none"
    vibe_str = ", ".join(prefs.vibe) if prefs.vibe else "classic"

    user_prompt = (
        f"City: {req.city}\n"
        f"Area: {prefs.area}\n"
        f"Duration: {prefs.duration_minutes} minutes\n"
        f"Budget: {prefs.budget}\n"
        f"Dietary restrictions: {dietary_str}\n"
        f"Group: {prefs.group_type}\n"
        f"Vibe: {vibe_str}\n"
        f"Walking level: {prefs.walking_level}\n"
        f"Time of day: {prefs.time_of_day}\n"
        f"Language: {prefs.language}\n"
        f"Extra notes: {prefs.constraints_note or 'none'}\n\n"
        "Generate a food tour itinerary."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        tour_data = json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Ensure stable IDs
    if not tour_data.get("tour_id"):
        tour_data["tour_id"] = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    tour_data.setdefault("assistant_context", {})["session_id"] = session_id

    # Persist
    tours = load_tours()
    tours.append({
        "tour_id": tour_data["tour_id"],
        "user_id": req.user_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": tour_data.get("title", "Rome food tour"),
        "description": tour_data.get("description", ""),
        "data": tour_data,
    })
    save_tours(tours)

    # Seed chat session with tour context
    SESSIONS[session_id] = {
        "tour_id": tour_data["tour_id"],
        "history": [],
        "tour_summary": user_prompt,
    }

    return tour_data


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = SESSIONS.setdefault(session_id, {"tour_id": req.tour_id, "history": [], "tour_summary": ""})

    system_msg = GUIDE_SYSTEM
    if session.get("tour_summary"):
        system_msg += f"\n\nCurrent tour context:\n{session['tour_summary']}"

    messages = [{"role": "system", "content": system_msg}]
    messages += session["history"][-10:]  # keep last 10 turns
    messages.append({"role": "user", "content": req.user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.5,
            messages=messages,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session["history"].append({"role": "user", "content": req.user_message})
    session["history"].append({"role": "assistant", "content": reply})

    return {"assistant_message": reply, "session_id": session_id}


@app.get("/api/tours")
def get_tours(user_id: str):
    tours = load_tours()
    user_tours = [t for t in tours if t.get("user_id") == user_id]
    # Return newest first, last 20
    return list(reversed(user_tours[-20:]))