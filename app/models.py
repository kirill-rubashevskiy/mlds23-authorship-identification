from sqlalchemy import Column, Integer

from app.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    requests = Column(Integer, default=0)
    rating = Column(Integer, default=None)
