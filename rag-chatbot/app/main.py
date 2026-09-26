import os
import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# Import modules
from app.schemas import LoginRequest, RegisterRequest, ChatRequest, ConfigUpdate, ConfigResponse
from app.models import User, ChatbotConfig
from app.core.security import verify_token
from app.core.database import engine, Base, get_db
from app.services import auth_service, llm_service, storage_service
from app.utils.text_processing import extract_text, chunk_text

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES CONFIGURATION ---

# 1. Mount the frontend folder for CSS/JS if needed (e.g., accessed via /static/script.js)
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# 2. Mount the App Static folder (for direct access to app/static files if needed later)
#    Accessed via: http://localhost:8001/assets/robot.jpg
if os.path.exists(os.path.join("app", "static")):
    app.mount("/assets", StaticFiles(directory=os.path.join("app", "static")), name="assets")

# --- IMAGE FIX FOR DASHBOARD ---
# This explicitly serves the images your Dashboard HTML is asking for (e.g., /robot.jpg)
# by fetching them from the app/static/ folder.
@app.get("/{image_name}.jpg")
async def serve_dashboard_images(image_name: str):
    # Define where your images are located
    base_image_path = os.path.join("app", "static")
    
    # Check if the requested image allows exists in that folder
    allowed_images = ["robot", "robo2", "robo3"] # Add more filenames here if needed
    
    if image_name in allowed_images:
        file_path = os.path.join(base_image_path, f"{image_name}.jpg")
        if os.path.exists(file_path):
            return FileResponse(file_path)
            
    return HTMLResponse("Image not found", status_code=404)

# --- HELPER: Get Config ---
def get_user_config(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    if not user: return None
    config = db.query(ChatbotConfig).filter(ChatbotConfig.user_id == user.id).first()
    if not config:
        config = ChatbotConfig(user_id=user.id)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config

# --- PAGE ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def root(): 
    return RedirectResponse(url="/login")

@app.get("/login")
async def serve_login():
    path = os.path.join("frontend", "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("login.html not found", status_code=404)

@app.get("/dashboard")
async def serve_dashboard():
    path = os.path.join("frontend", "dashboard.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("dashboard.html not found", status_code=404)

@app.get("/integration")
async def serve_embed():
    path = os.path.join("frontend", "embed.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("embed.html not found", status_code=404)

# --- CONFIG ROUTES ---
@app.get("/api/config", response_model=ConfigResponse)
async def get_config(username: str = Depends(verify_token), db: Session = Depends(get_db)):
    config = get_user_config(db, username)
    return config

@app.post("/api/config")
async def update_config(req: ConfigUpdate, username: str = Depends(verify_token), db: Session = Depends(get_db)):
    config = get_user_config(db, username)
    for key, value in req.dict().items():
        setattr(config, key, value)
    db.commit()
    return {"message": "Configuration saved"}

@app.get("/api/public/config")
async def get_public_config(username: str, db: Session = Depends(get_db)):
    config = get_user_config(db, username)
    if not config: 
        return {"bot_name": "Assistant", "primary_color": "#000", "text_color": "#fff", "font_family": "Inter", "chat_shape": "circle", "chat_icon": "message"}
    return {
        "bot_name": config.bot_name, "primary_color": config.primary_color, "text_color": config.text_color,
        "font_family": config.font_family, "chat_shape": config.chat_shape, "chat_icon": config.chat_icon
    }

# --- AUTH ROUTES ---
@app.post("/api/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if len(req.username) < 3: raise HTTPException(400, "Username too short")
    if not auth_service.register_user(db, req.username, req.password): 
        raise HTTPException(400, "User exists")
    user = db.query(User).filter(User.username == req.username).first()
    if not db.query(ChatbotConfig).filter(ChatbotConfig.user_id == user.id).first():
        db.add(ChatbotConfig(user_id=user.id))
        db.commit()
    return {"token": auth_service.create_access_token(req.username), "username": req.username}

@app.post("/api/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    if auth_service.authenticate_user(db, req.username, req.password):
        return {"token": auth_service.create_access_token(req.username), "username": req.username}
    raise HTTPException(401, "Invalid credentials")

# --- DOCUMENT ROUTES ---
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), username: str = Depends(verify_token), db: Session = Depends(get_db)):
    content = await file.read()
    text = extract_text(content, file.filename)
    if not text: raise HTTPException(400, "No text found")
    chunks = chunk_text(text)
    if not chunks: raise HTTPException(400, "File empty")
    embeddings = np.array(llm_service.get_embedding(chunks)).astype('float32')
    count = storage_service.add_document(db, username, file.filename, chunks, embeddings)
    return {"message": "Uploaded", "chunks": count, "filename": file.filename}

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, username: str = Depends(verify_token), db: Session = Depends(get_db)):
    storage_service.delete_document(db, username, filename)
    return {"message": "Deleted"}

@app.get("/api/documents")
async def get_documents(username: str = Depends(verify_token), db: Session = Depends(get_db)):
    return {"documents": storage_service.get_user_documents(db, username)}

# --- CHAT ROUTES ---
@app.post("/api/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_token), db: Session = Depends(get_db)):
    return await process_chat(req.message, username, db)

@app.post("/api/widget/chat")
async def widget_chat(req: ChatRequest, username: str, db: Session = Depends(get_db)):
    return await process_chat(req.message, username, db)

# --- INTELLIGENT CHAT PROCESSING ---
async def process_chat(message: str, username: str, db: Session):
    index, chunks = storage_service.load_user_session(username, db)
    
    context_text = ""
    if chunks and len(chunks) > 0:
        query_vec = np.array(llm_service.get_embedding([message])).astype('float32')
        k = min(3, len(chunks))
        k = min(k, index.ntotal)
        if k > 0:
            _, indices = index.search(query_vec, k)
            relevant_docs = [chunks[i] for i in indices[0] if i < len(chunks) and i != -1]
            if relevant_docs:
                context_text = "\n\n".join(relevant_docs)
    
    prompt = f"""
    You are Zephra, a warm, energetic, and professional AI assistant.
    
    --- STRICT GUIDELINES ---
    1. **NO META-TALK**: NEVER mention "uploaded documents", "files", "context", or "this website".
    2. **VARIETY**: Do not use the same greeting twice.
    3. **EAGERNESS**: Always sound happy to help.

    --- BACKGROUND INFO (Internal Knowledge) ---
    {context_text}

    --- USER MESSAGE ---
    {message}

    --- YOUR RESPONSE ---
    """
    
    try:
        response_text = llm_service.get_llm_response(prompt)
        return {"response": response_text, "sources": []}
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"response": "I'm having a brief connection moment. Please try again in a second.", "sources": []}

# --- WIDGET JS ---
@app.get("/widget.js")
async def get_widget_js(request: Request):
    base_url = str(request.base_url).rstrip('/')
    js_code = f"""
(function() {{
    const scriptTag = document.currentScript || document.getElementById('zephra-chat-script');
    const username = scriptTag ? scriptTag.getAttribute('data-username') : null;
    const apiUrl = "{base_url}";

    if (!username) {{ console.error("Zephra Widget: data-username attribute missing"); return; }}

    const fontLink = document.createElement('link');
    fontLink.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap";
    fontLink.rel = "stylesheet";
    document.head.appendChild(fontLink);

    const icons = {{
        message: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>',
        bot: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path></svg>',
        sparkle: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>'
    }};

    const style = document.createElement('style');
    document.head.appendChild(style);

    const bubble = document.createElement('div'); bubble.id = "zephra-bubble";
    document.body.appendChild(bubble);

    const chatBox = document.createElement('div'); chatBox.id = "zephra-window";
    chatBox.innerHTML = `
        <div class="zephra-header">
            <span id="zephra-header-title">Assistant</span>
            <span id="zephra-close" style="cursor:pointer;font-size:24px;">&times;</span>
        </div>
        <div class="zephra-messages" id="zephra-messages"><div class="z-msg z-bot">Hello! How can I help you today?</div></div>
        <div id="zephra-typing" style="display:none; padding:10px; font-size:12px; color:#666;">AI is thinking...</div>
        <div class="zephra-input-area">
            <input id="zephra-input" type="text" placeholder="Ask a question...">
            <button id="zephra-send">Send</button>
        </div>
    `;
    document.body.appendChild(chatBox);

    fetch(`${{apiUrl}}/api/public/config?username=${{username}}`)
        .then(r => r.json())
        .then(config => {{
            let radius = config.chat_shape === 'square' ? '8px' : (config.chat_shape === 'squircle' ? '18px' : '50%');
            bubble.innerHTML = icons[config.chat_icon] || icons.message;
            document.getElementById('zephra-header-title').innerText = config.bot_name;

            style.innerHTML = `
                #zephra-bubble {{ background:${{config.primary_color}}; color:${{config.text_color}}; border-radius:${{radius}}; position:fixed; bottom:20px; right:20px; width:60px; height:60px; cursor:pointer; display:flex; align-items:center; justify-content:center; z-index:999999; box-shadow:0 4px 12px rgba(0,0,0,0.2); transition: transform 0.2s; }}
                #zephra-bubble:hover {{ transform: scale(1.1); }}
                #zephra-window {{ font-family:'${{config.font_family}}', sans-serif; position:fixed; bottom:90px; right:20px; width:350px; height:500px; background:white; border-radius:12px; box-shadow:0 8px 30px rgba(0,0,0,0.1); display:none; flex-direction:column; z-index:999999; overflow:hidden; border: 1px solid #eee; }}
                .zephra-header {{ background:${{config.primary_color}}; color:${{config.text_color}}; padding:15px; display:flex; justify-content:space-between; align-items:center; font-weight:600; }}
                .zephra-messages {{ flex:1; padding:15px; overflow-y:auto; background:#f9f9f9; display:flex; flex-direction:column; gap:10px; }}
                .zephra-input-area {{ padding:15px; display:flex; gap:10px; border-top:1px solid #eee; }}
                #zephra-input {{ flex:1; padding:8px; border:1px solid #ddd; border-radius:4px; outline:none; }}
                #zephra-send {{ background:${{config.primary_color}}; color:${{config.text_color}}; border:none; padding:8px 15px; border-radius:4px; cursor:pointer; font-weight:600; }}
                .z-msg {{ max-width:80%; padding:10px; border-radius:8px; font-size:14px; line-height:1.4; }}
                .z-user {{ align-self:flex-end; background:${{config.primary_color}}; color:${{config.text_color}}; }}
                .z-bot {{ align-self:flex-start; background:#eee; color:#333; }}
            `;
        }});

    bubble.onclick = () => chatBox.style.display = chatBox.style.display === 'flex' ? 'none' : 'flex';
    document.getElementById('zephra-close').onclick = () => chatBox.style.display = 'none';

    const sendMessage = async () => {{
        const input = document.getElementById('zephra-input');
        const text = input.value.trim();
        if (!text) return;

        const msgs = document.getElementById('zephra-messages');
        const typing = document.getElementById('zephra-typing');
        
        const uDiv = document.createElement('div'); uDiv.className = 'z-msg z-user'; uDiv.textContent = text;
        msgs.appendChild(uDiv);
        input.value = '';
        msgs.scrollTop = msgs.scrollHeight;

        typing.style.display = 'block';
        try {{
            const res = await fetch(`${{apiUrl}}/api/widget/chat?username=${{username}}`, {{
                method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ message: text }})
            }});
            const data = await res.json();
            typing.style.display = 'none';
            const bDiv = document.createElement('div'); bDiv.className = 'z-msg z-bot'; bDiv.textContent = data.response;
            msgs.appendChild(bDiv);
            msgs.scrollTop = msgs.scrollHeight;
        }} catch (e) {{ typing.style.display = 'none'; }}
    }};

    document.getElementById('zephra-send').onclick = sendMessage;
    document.getElementById('zephra-input').onkeypress = (e) => {{ if(e.key==='Enter') sendMessage(); }};
}})();
    """
    return Response(content=js_code, media_type="application/javascript")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)