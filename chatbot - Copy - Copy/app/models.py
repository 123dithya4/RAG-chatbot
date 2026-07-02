from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="owner")
    config = relationship("ChatbotConfig", back_populates="owner", uselist=False)
    vector_store = relationship("UserVectorStore", back_populates="owner", uselist=False)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="documents")

class ChatbotConfig(Base):
    __tablename__ = "chatbot_configs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    
    bot_name = Column(String, default="AI Assistant")
    primary_color = Column(String, default="#000000")
    text_color = Column(String, default="#FFFFFF")
    font_family = Column(String, default="Inter")
    chat_shape = Column(String, default="circle")
    chat_icon = Column(String, default="message")
    
    owner = relationship("User", back_populates="config")

# --- NEW TABLE FOR BINARY DATA ---
class UserVectorStore(Base):
    __tablename__ = "user_vector_stores"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    
    # We store the FAISS index and Chunks as raw binary data (BLOBs)
    faiss_index = Column(LargeBinary) 
    chunks = Column(LargeBinary)
    
    owner = relationship("User", back_populates="vector_store")