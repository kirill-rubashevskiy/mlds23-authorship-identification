from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    """
    Function adds new user to db.

    :param db: sqlalchemy.orm.Session
    :param user: user to add to db
    :return: added user
    """
    db_user = models.User(**user.model_dump())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user_requests(db: Session, user_id: int) -> models.User:
    """
    Function updates in db number of requests made my user.

    :param db: sqlalchemy.orm.Session
    :param user_id: user id
    :return: updated user
    """
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    db_user.requests += 1
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user_rating(db: Session, user_id: int, rating: int) -> models.User:
    """
    Function updates in db rating, given to the service by user.

    :param db: sqlalchemy.orm.Session
    :param user_id: user id
    :param rating: rating
    :return: updated user
    """
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    db_user.rating = rating
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user(db: Session, user_id: int) -> models.User | None:
    """
    Function searches user in db by id.

    :param db: sqlalchemy.orm.Session
    :param user_id: user id
    :return: search results
    """
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_total_users(db: Session) -> int:
    """
    Function calculates total number of the service users.

    :param db: sqlalchemy.orm.Session
    :return: total number of users
    """
    return db.query(func.count(models.User.id)).scalar()


def get_total_requests(db: Session) -> int:
    """
    Function calculates total number of requests to the service made by the users.

    :param db: sqlalchemy.orm.Session
    :return: total number of requests
    """
    total_requests = db.query(func.sum(models.User.requests)).scalar()
    if total_requests is None:
        return 0
    return total_requests


def get_avg_rating(db: Session) -> int | float:
    """
    Function calculates average rating given to the service by users.
    If no ratings were given, it returns 0.

    :param db: sqlalchemy.orm.Session
    :return: average rating or 0
    """
    total_ratings = db.query(func.count(models.User.rating)).scalar()
    if total_ratings == 0:
        return 0
    else:
        return db.query(func.sum(models.User.rating)).scalar() / total_ratings
