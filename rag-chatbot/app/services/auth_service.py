from sqlalchemy.orm import Session
from app.models import User
import hashlib
import jwt
from datetime import datetime, timedelta
from app.core.config import SECRET_KEY, ALGORITHM

# Verify password
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    if user.password_hash == pwd_hash:
        return True
    return False

# Create new user
def register_user(db: Session, username: str, password: str):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return False
    
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    
    new_user = User(username=username, password_hash=pwd_hash)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return True

# Generate Token
def create_access_token(username: str):
    expire = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, ALGORITHM)