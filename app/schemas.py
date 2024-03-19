from pydantic import BaseModel


class Prediction(BaseModel):
    label: int
    name: str


class Stats(BaseModel):
    total_users: int
    total_requests: int
    avg_rating: int | float


class Text(BaseModel):
    text: str


class UserCreate(BaseModel):
    id: int

    class Config:
        from_attributes = True


class User(UserCreate):
    requests: int = 0
    rating: int | None = None

    class Config:
        from_attributes = True
