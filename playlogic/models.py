from pydantic import BaseModel

class Observation(BaseModel):
    text: str

class Action(BaseModel):
    text: str