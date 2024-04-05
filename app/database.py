from sqlalchemy.orm import declarative_base, sessionmaker


SessionLocal = sessionmaker(autocommit=False, autoflush=False)

Base = declarative_base()
