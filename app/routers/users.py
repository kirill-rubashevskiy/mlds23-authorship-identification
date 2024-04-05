import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy.orm import Session

from app import crud, schemas
from app.dependencies import get_db, my_key_builder


router = APIRouter(prefix="/users", tags=["users"])


@router.get("/")
def root():
    return {"message": "Welcome to the Authorship Identification Service"}


@router.post("/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Function checks if user already exists and if not — creates him.

    :param user: user to create
    :param db: sqlalchemy.orm.Session
    :return: created user on message that user already exists
    """
    db_user = crud.get_user(db, user_id=user.id)
    if db_user:
        raise HTTPException(status_code=400, detail="User already registered")
    return crud.create_user(db=db, user=user)


@router.patch("/{user_id}/requests", response_model=schemas.User)
def update_user_requests(user_id: int, db: Session = Depends(get_db)):
    """
    Function updates number of requests made my user.
    :param db: sqlalchemy.orm.Session
    :param user_id: user id
    :return: updated user
    """

    db_user = crud.get_user(db=db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.update_user_requests(db=db, user_id=user_id)


@router.patch("/{user_id}/{rating}", response_model=schemas.User)
def update_user_rating(user_id: int, rating: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db=db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.update_user_rating(db=db, user_id=user_id, rating=rating)


@router.get("/stats", response_model=schemas.Stats)
@cache(expire=30, key_builder=my_key_builder)
def get_stats(db: Session = Depends(get_db)):
    """
    Function returns bot usage statistics.

    :param db: sqlalchemy.orm.Session
    :return: bot statistics.
    """
    total_users = crud.get_total_users(db=db)
    total_requests = crud.get_total_requests(db=db)
    avg_rating = np.round(crud.get_avg_rating(db=db), 2)
    return schemas.Stats(
        total_users=total_users, total_requests=total_requests, avg_rating=avg_rating
    )
