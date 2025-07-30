import os
from sqlalchemy import create_engine
from sqlalchemy import Column, String
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from __init__ import DB_DIR, DATABASE_URL
from sqlalchemy.ext.declarative import declarative_base

os.makedirs(DB_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        init_admin_user(db)
    finally:
        db.close()


def init_admin_user(db):
    admin_user = db.query(User).filter(User.username == "admin").first()
    if not admin_user:
        hashed_password = CryptContext(schemes=["bcrypt"], deprecated="auto").hash("6QVnYsC4iSzz")
        db.add(User(username="admin", hashed_password=hashed_password))
        db.commit()


class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)
