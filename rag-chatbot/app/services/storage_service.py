import faiss
import pickle
import numpy as np
from typing import Dict, Tuple
from sqlalchemy.orm import Session
from app.models import User, Document, UserVectorStore
from app.core.config import EMBEDDING_DIM

# In-memory cache to prevent constant DB reads during a chat session
# Structure: { "username": (index_object, chunks_list) }
memory_cache: Dict[str, Tuple[faiss.Index, list]] = {}

def faiss_to_bytes(index) -> bytes:
    """Convert FAISS index to bytes for DB storage"""
    # serialize_index returns a numpy array of uint8
    return faiss.serialize_index(index).tobytes()

def bytes_to_faiss(data: bytes):
    """Convert bytes back to FAISS index"""
    # Convert bytes back to numpy uint8 array
    vector_array = np.frombuffer(data, dtype='uint8')
    return faiss.deserialize_index(vector_array)

def load_user_session(username: str, db: Session):
    """Loads FAISS index and Chunks from Cache or DB."""
    # 1. Check RAM Cache first
    if username in memory_cache:
        return memory_cache[username]

    # 2. Check Database
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return faiss.IndexFlatL2(EMBEDDING_DIM), []

    vector_store = db.query(UserVectorStore).filter(UserVectorStore.user_id == user.id).first()

    if vector_store and vector_store.faiss_index and vector_store.chunks:
        try:
            index = bytes_to_faiss(vector_store.faiss_index)
            chunks = pickle.loads(vector_store.chunks)
        except Exception as e:
            print(f"Error loading binary data: {e}")
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            chunks = []
    else:
        # Create fresh if not exists
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        chunks = []

    # Save to RAM cache
    memory_cache[username] = (index, chunks)
    return index, chunks

def save_user_session(username: str, db: Session, index, chunks):
    """Saves memory to DB as Binary."""
    user = db.query(User).filter(User.username == username).first()
    if not user: return

    # Update Cache
    memory_cache[username] = (index, chunks)

    # Serialize Data
    binary_index = faiss_to_bytes(index)
    binary_chunks = pickle.dumps(chunks)

    # Check if record exists
    store_record = db.query(UserVectorStore).filter(UserVectorStore.user_id == user.id).first()

    if store_record:
        store_record.faiss_index = binary_index
        store_record.chunks = binary_chunks
    else:
        new_record = UserVectorStore(
            user_id=user.id,
            faiss_index=binary_index,
            chunks=binary_chunks
        )
        db.add(new_record)
    
    db.commit()

def add_document(db: Session, username: str, filename: str, new_chunks: list, embeddings):
    # 1. Update Documents Table
    user = db.query(User).filter(User.username == username).first()
    db.add(Document(filename=filename, user_id=user.id))
    db.commit()

    # 2. Load Current State
    index, chunks = load_user_session(username, db)
    
    # 3. Add New Data
    index.add(embeddings)
    chunks.extend(new_chunks)

    # 4. Save Back to DB
    save_user_session(username, db, index, chunks)
    return len(new_chunks)

def delete_document(db: Session, username: str, filename: str):
    # 1. Remove from Documents Table
    user = db.query(User).filter(User.username == username).first()
    doc = db.query(Document).filter(Document.filename == filename, Document.user_id == user.id).first()
    
    if doc:
        db.delete(doc)
        db.commit()
    
    # 2. Reset Vector Memory (Simple approach: Clear all)
    # Since we store vectors as a blob, selectively deleting is hard. 
    # We clear the index so the user doesn't get results from deleted files.
    new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    new_chunks = []
    
    save_user_session(username, db, new_index, new_chunks)
    return True

def get_user_documents(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    return [doc.filename for doc in user.documents] if user else []