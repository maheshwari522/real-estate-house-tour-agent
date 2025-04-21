from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str

class Message(BaseModel):
    question: str
    answer: str
