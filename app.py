from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
from transformers import pipeline

app = FastAPI()

MODEL_NAME = "unitary/toxic-bert"
guardrail_model = pipeline("text-classification", model=MODEL_NAME)
sessions = {}  # session_id: policy

class StartSessionRequest(BaseModel):
    policy: str
    user_id: str

class StartSessionResponse(BaseModel):
    session_id: str

class CheckMessageRequest(BaseModel):
    session_id: str
    rail: str
    message: str

@app.post("/start_session", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest):
    session_id = str(uuid4())
    sessions[session_id] = req.policy
    return {"session_id": session_id}

@app.post("/check_message")
async def check_message(req: CheckMessageRequest):
    policy = sessions.get(req.session_id)
    if not policy:
        return {"error": "Invalid session_id."}
    result = guardrail_model(req.message)
    label = result[0]['label']
    score = result[0]['score']
    return {
        "rail": req.rail,
        "label": label,
        "score": score,
        "policy": policy
    }
