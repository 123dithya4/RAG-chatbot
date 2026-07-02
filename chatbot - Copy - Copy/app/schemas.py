from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str

# UPDATED: Includes Font, Shape, and Icon
class ConfigUpdate(BaseModel):
    bot_name: str
    primary_color: str
    text_color: str
    font_family: str
    chat_shape: str
    chat_icon: str

class ConfigResponse(BaseModel):
    bot_name: str
    primary_color: str
    text_color: str
    font_family: str
    chat_shape: str
    chat_icon: str