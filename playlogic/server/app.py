from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import json
from openenv import create_app
from playlogic.server.environment import MultiAgentFootball
import numpy as np

env = MultiAgentFootball()
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/reset")
async def reset():
    obs = env.reset()
    return {"observation": obs.text, "state": env.state()}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action_text = data["action"]
    obs, rewards, done, info = env.step(Action(text=action_text))
    return {"observation": obs.text, "reward": rewards, "done": done, "info": info, "state": env.state()}