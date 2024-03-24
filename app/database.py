from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


SessionLocal = sessionmaker(autocommit=False, autoflush=False)

Base = declarative_base()
