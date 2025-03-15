from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# application import config.
from core.config import db

# DB URL for connection
SQLALCHEMY_DATABASE_URL = db.DATABASE_URL

# Creating DB engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Creating and Managing session.
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Domain Modelling Dependency
Base = declarative_base()

print("Database is Ready!")
