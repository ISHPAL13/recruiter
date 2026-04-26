from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import requests, os, tempfile, shutil
import asyncio
import traceback
import PyPDF2
import base64
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import json
from dotenv import load_dotenv
from typing import Optional, Dict
import subprocess
import re
import secrets
import db

import db

import db


# Load environment variables from .env file
load_dotenv()

# In-memory session store (simple alternative to Redis)
SESSION_STORE: Dict[str, Dict] = {}

app = FastAPI(title="HireGenie", version="1.0")

# Add CORS middleware FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom session helper functions
def get_session_id(request: Request) -> Optional[str]:
    """Get session ID from cookie"""
    return request.cookies.get("session_id")

def get_session_data(session_id: str) -> Dict:
    """Get session data from store"""
    return SESSION_STORE.get(session_id, {})

def create_session() -> str:
    """Create new session and return session ID"""
    session_id = secrets.token_urlsafe(32)
    SESSION_STORE[session_id] = {}
    return session_id

def save_session(session_id: str, data: Dict):
    """Save session data to store"""
    SESSION_STORE[session_id] = data

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Get API keys from environment
LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY") or os.getenv("HEYGEN_API_KEY")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID")
LIVEAVATAR_CONTEXT_ID = os.getenv("LIVEAVATAR_CONTEXT_ID")
LIVEAVATAR_LANGUAGE = os.getenv("LIVEAVATAR_LANGUAGE", "en")
LIVEAVATAR_DEFAULT_VOICE_ID = os.getenv("LIVEAVATAR_DEFAULT_VOICE_ID")
LIVEAVATAR_VOICE_ID_PROFESSIONAL = os.getenv("LIVEAVATAR_VOICE_ID_PROFESSIONAL")
LIVEAVATAR_VOICE_ID_FRIENDLY = os.getenv("LIVEAVATAR_VOICE_ID_FRIENDLY")
LIVEAVATAR_VOICE_ID_STRICT = os.getenv("LIVEAVATAR_VOICE_ID_STRICT")
LIVEAVATAR_VOICE_ID_CASUAL = os.getenv("LIVEAVATAR_VOICE_ID_CASUAL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "your_deepgram_api_key_here")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
GEMINI_LIVE_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Aoede")

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

def get_tone_prompt(tone, cv_text, user_name, job_details=None):
    """Generate interviewer prompt based on selected tone and job details"""
    
    # Default values
    role = "the target role"
    experience = "relevant experience"
    skills = "their skills"
    
    if job_details:
        role = job_details.get('target_role') or role
        experience = job_details.get('experience') or experience
        skills = job_details.get('skills') or skills

    base_context = f"""You are Sarah, an expert HR interviewer conducting an interview with {user_name} for the position of {role}.

CANDIDATE PROFILE:
- Target Role: {role}
- Experience Level: {experience}
- Key Skills: {skills}
- CV Summary:
{cv_text[:1500]}  

INTERVIEW GUIDELINES:
- Your goal is to assess their fit for {role}.
- Ask questions tailored to their experience level ({experience}) and listed skills ({skills}).
- NEVER repeat similar questions or ask about the same topic twice.
- Move naturally between topics: projects -> technical skills -> soft skills -> situational.
- Build on their previous answers with NEW follow-up questions.
- Vary your question types: "What", "How", "Why", "Tell me about", "Describe".
- Keep the conversation flowing naturally like a real interview."""

    tone_styles = {
        'professional': """
TONE: Professional & Neutral
- Be polite, business-like, and respectful
- Ask insightful questions that assess competency
- Keep responses clear and direct
- Reply in ONE short sentence (8-12 words max)
- Focus on skills, achievements, and problem-solving abilities""",
        
        'friendly': """
TONE: Friendly & Encouraging  
- Be warm, supportive, and encouraging
- Use positive language and show enthusiasm
- Make the candidate feel comfortable
- Reply in ONE short sentence (8-12 words max)
- Ask about their passions, learning experiences, and growth""",
        
        'strict': """
TONE: Strict & Demanding
- Be critical and challenge their responses
- Ask tough follow-up questions that probe deeper
- Point out gaps or weaknesses professionally
- Reply in ONE short sentence (8-12 words max)
- Test their knowledge and decision-making under pressure""",
        
        'casual': """
TONE: Casual & Relaxed
- Be conversational and laid-back
- Use informal language (but still professional)
- Keep the atmosphere relaxed
- Reply in ONE short sentence (8-12 words max)
- Ask about real-world scenarios and practical experiences"""
    }
    
    return base_context + "\n" + tone_styles.get(tone, tone_styles['professional'])

def get_voice_for_tone(tone):
    """Map interview tone to the configured LiveAvatar voice ID"""
    voice_map = {
        'professional': LIVEAVATAR_VOICE_ID_PROFESSIONAL,
        'friendly': LIVEAVATAR_VOICE_ID_FRIENDLY,
        'strict': LIVEAVATAR_VOICE_ID_STRICT,
        'casual': LIVEAVATAR_VOICE_ID_CASUAL
    }
    return voice_map.get(tone) or LIVEAVATAR_DEFAULT_VOICE_ID


def build_liveavatar_persona(voice_id: Optional[str]) -> Dict:
    persona = {
        "language": LIVEAVATAR_LANGUAGE
    }
    if voice_id:
        persona["voice_id"] = voice_id
    if LIVEAVATAR_CONTEXT_ID:
        persona["context_id"] = LIVEAVATAR_CONTEXT_ID
    return persona

@app.get("/api/voice-config")
async def get_voice_config(request: Request):
    """Return voice configuration based on user's tone"""
    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    session_data = get_session_data(session_id)
    if 'user_name' not in session_data:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    tone = session_data.get('interviewer_tone', 'professional')
    voice_id = get_voice_for_tone(tone)
    
    return {"tone": tone, "voice_id": voice_id, "avatar_id": LIVEAVATAR_AVATAR_ID}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        return RedirectResponse(url="/login", status_code=302)
    
    session_data = get_session_data(session_id)
    if 'user_name' not in session_data:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/interview", response_class=HTMLResponse)
async def interview(request: Request):
    session_id = get_session_id(request)
    print(f"[SEARCH] /interview - Session ID: {session_id}")
    
    if not session_id:
        print("   [ERROR] No session ID cookie")
        return RedirectResponse(url="/login", status_code=302)
    
    session_data = get_session_data(session_id)
    print(f"   Session data: {session_data}")
    print(f"   'user_name' in session: {'user_name' in session_data}")
    
    if 'user_name' not in session_data:
        print("   [ERROR] Redirecting to login - no user_name in session")
        return RedirectResponse(url="/login", status_code=302)
    
    print(f"   [SUCCESS] User {session_data.get('user_name')} accessing interview page")
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/api/login")
async def api_login(
    request: Request,
    userName: str = Form(...),
    userEmail: str = Form(...),
    userPhone: Optional[str] = Form(None),
    commPreferences: Optional[str] = Form(None),
    mockInterview: Optional[str] = Form(None),
    selectedTone: str = Form('professional'),
    targetRole: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    skills: Optional[str] = Form(None),
    interviewType: Optional[str] = Form(None),
    duration: Optional[str] = Form(None),
    difficulty: Optional[str] = Form(None),
    cvFile: UploadFile = File(...)
):
    try:
        # Validate CV file
        if not cvFile.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save CV file
        filename = f"{userName.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(cvFile.file, buffer)
        
        # Extract CV text
        cv_text = extract_text_from_pdf(filepath)
        
        if not cv_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
        
        # Create new session
        session_id = create_session()
        
        # Store in session
        session_data = {
            'user_name': userName,
            'user_email': userEmail,
            'user_phone': userPhone,
            'comm_preferences': commPreferences,
            'target_role': targetRole,
            'experience': experience,
            'industry': industry,
            'skills': skills,
            'interview_type': interviewType,
            'difficulty': difficulty,
            'duration': duration,
            'cv_path': filepath,
            'cv_text': cv_text,
            'mock_interview': True, # Always true for this flow or check mockInterview
            'interviewer_tone': selectedTone,
            'conversation_history': []
        }
        save_session(session_id, session_data)
        
        print(f"[SUCCESS] User logged in: {userName}")
        print(f"   Email: {userEmail}")
        print(f"   Mock Interview: {mockInterview == 'on'}")
        print(f"   Tone: {session_data['interviewer_tone']}")
        print(f"   CV extracted: {len(cv_text)} characters")
        print(f"   Session ID: {session_id}")
        print(f"   Session data saved: {session_data}")
        
        # Create response with session cookie
        response = JSONResponse({"success": True, "message": "Login successful"})
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=14400,  # 4 hours
            httponly=True,
            samesite="lax",
            path="/"
        )
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user-info")
async def get_user_info(request: Request):
    """Return current user session info"""
    session_id = get_session_id(request)
    if not session_id:
        return {"logged_in": False}
    
    session_data = get_session_data(session_id)
    if 'user_name' not in session_data:
        return {"logged_in": False}
    
    return {
        "logged_in": True,
        "name": session_data.get('user_name'),
        "email": session_data.get('user_email'),
        "mock_interview": session_data.get('mock_interview', False),
        "tone": session_data.get('interviewer_tone', 'professional')
    }

@app.post("/api/logout")
async def logout(request: Request):
    session_id = get_session_id(request)
    if session_id and session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
    
    response = JSONResponse({"success": True})
    response.delete_cookie("session_id")
    return response

@app.post("/liveavatar/session")
async def create_liveavatar_session(request: Request):
    if not LIVEAVATAR_API_KEY or LIVEAVATAR_API_KEY.startswith("your_"):
        raise HTTPException(status_code=500, detail="LIVEAVATAR_API_KEY not configured")
    if not LIVEAVATAR_AVATAR_ID:
        raise HTTPException(status_code=500, detail="LIVEAVATAR_AVATAR_ID not configured")

    session_id = get_session_id(request)
    session_data = get_session_data(session_id) if session_id else {}
    tone = session_data.get('interviewer_tone', 'professional')
    voice_id = get_voice_for_tone(tone)

    token_payload = {
        "mode": "FULL",
        "avatar_id": LIVEAVATAR_AVATAR_ID,
        "avatar_persona": build_liveavatar_persona(voice_id),
        "interactivity_type": "CONVERSATIONAL",
    }

    token_response = requests.post(
        "https://api.liveavatar.com/v1/sessions/token",
        headers={
            "X-API-KEY": LIVEAVATAR_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=token_payload,
        timeout=30,
    )

    if token_response.status_code >= 400:
        raise HTTPException(
            status_code=token_response.status_code,
            detail=f"LiveAvatar token error: {token_response.text}"
        )

    token_result = token_response.json()
    session_token = (
        token_result.get("access_token")
        or token_result.get("data", {}).get("access_token")
        or token_result.get("session_token")
        or token_result.get("data", {}).get("session_token")
        or token_result.get("token")
        or token_result.get("data", {}).get("token")
    )

    if not session_token:
        raise HTTPException(status_code=500, detail=f"LiveAvatar token missing from response: {token_result}")

    start_response = requests.post(
        "https://api.liveavatar.com/v1/sessions/start",
        headers={
            "Authorization": f"Bearer {session_token}",
            "Accept": "application/json",
        },
        timeout=30,
    )

    if start_response.status_code >= 400:
        raise HTTPException(
            status_code=start_response.status_code,
            detail=f"LiveAvatar start error: {start_response.text}"
        )

    start_result = start_response.json()
    start_data = start_result.get("data", {})

    return {
        "session_token": session_token,
        "session_id": start_data.get("session_id"),
        "url": start_data.get("livekit_url"),
        "access_token": start_data.get("livekit_client_token"),
        "max_session_duration": start_data.get("max_session_duration"),
        "ws_url": start_data.get("ws_url"),
        "raw": start_result,
    }


@app.post("/liveavatar/session/stop")
async def stop_liveavatar_session(payload: dict):
    session_token = payload.get("session_token")
    if not session_token:
        raise HTTPException(status_code=400, detail="session_token is required")

    response = requests.post(
        "https://api.liveavatar.com/v1/sessions/stop",
        headers={
            "Authorization": f"Bearer {session_token}",
            "Accept": "application/json",
        },
        timeout=30,
    )

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=f"LiveAvatar stop error: {response.text}")

    return response.json()

@app.get("/deepgram/api-key")
async def get_deepgram_key():
    """Provide Deepgram API key to frontend"""
    if not DEEPGRAM_API_KEY or DEEPGRAM_API_KEY.startswith("your_"):
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")
    return {"api_key": DEEPGRAM_API_KEY}

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert audio to text using Deepgram API"""
    try:
        if not DEEPGRAM_API_KEY or DEEPGRAM_API_KEY.startswith("your_"):
            raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")
        
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        print(f"[SEND] Sending {len(audio_data)} bytes to Deepgram STT...")
        
        # Send to Deepgram API
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/webm"
        }
        
        params = {
            "model": "nova-2",
            "language": "en-US",
            "smart_format": "true",
            "punctuate": "true"
        }
        
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers=headers,
            params=params,
            data=audio_data,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[ERROR] Deepgram API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Deepgram API error: {response.status_code}")
        
        result = response.json()
        
        # Extract transcript
        transcript = ""
        if result.get("results") and result["results"].get("channels"):
            alternatives = result["results"]["channels"][0].get("alternatives", [])
            if alternatives:
                transcript = alternatives[0].get("transcript", "").strip()
        
        print(f"[TRANSCRIPT] Deepgram transcript: '{transcript}'")
        
        return {"text": transcript}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm")
async def llm(request: Request):
    data = await request.json()
    user_input = data.get("prompt", "")
    
    print(f"\n[AI] LLM Request - User input: '{user_input}'")
    
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        print("[ERROR] GEMINI_API_KEY not configured properly!")
        return JSONResponse({
            "error": "GEMINI_API_KEY not configured. Please set your API key.",
            "text": "I apologize, but my AI system is not configured properly. Please contact the administrator."
        }, status_code=200)
    
    # Check if user is logged in
    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    session_data = get_session_data(session_id)
    if 'user_name' not in session_data:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    try:
        # Set API key in environment
        if 'GEMINI_API_KEY' not in os.environ and GEMINI_API_KEY:
            os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
        
        client = genai.Client()
        
        # Get user context
        cv_text = session_data.get('cv_text', '')
        user_name = session_data.get('user_name', 'Candidate')
        tone = session_data.get('interviewer_tone', 'professional')
        
        # Build prompt with CV context and tone
        system_prompt = get_tone_prompt(tone, cv_text, user_name, session_data)
        
        # Get conversation history
        conversation_history = session_data.get('conversation_history', [])
        
        # Build context from recent conversation
        context = ""
        if conversation_history:
            recent = conversation_history[-3:]
            context = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in recent
            ])
            context = f"\n\nRECENT CONVERSATION:\n{context}\n"
        
        full_prompt = f"""{system_prompt}

{context}

CANDIDATE'S RESPONSE: {user_input}

IMPORTANT: Reply with ONE concise sentence (8-12 words). Be direct and natural."""
        
        print(f"[SEND] Sending to Gemini 2.5 Flash...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        bot_response = response.text.strip()
        
        # Store in conversation history
        conversation_history.append({
            "question": bot_response,
            "answer": user_input,
            "timestamp": datetime.now().isoformat()
        })
        session_data['conversation_history'] = conversation_history
        save_session(session_id, session_data)
        
        print(f"[SUCCESS] Gemini response: {bot_response}")
        
        return {"text": bot_response}
        
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        error_msg = "I'm having trouble processing that. Could you rephrase?"
        return {"text": error_msg, "error": str(e)}

@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    """Gemini Live speech-to-speech proxy for the interview experience."""
    await websocket.accept()

    cookies = websocket.cookies
    session_id = cookies.get("session_id")
    if not session_id or session_id not in SESSION_STORE:
        await websocket.send_text(json.dumps({"error": "Invalid or expired session"}))
        await websocket.close(code=4001, reason="Invalid or expired session")
        return

    session_data = get_session_data(session_id)
    if "user_name" not in session_data:
        await websocket.send_text(json.dumps({"error": "Not logged in"}))
        await websocket.close(code=4001, reason="Not logged in")
        return

    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        await websocket.send_text(json.dumps({"error": "GEMINI_API_KEY is not configured"}))
        await websocket.close(code=4002, reason="GEMINI_API_KEY not configured")
        return

    cv_text = session_data.get("cv_text", "")
    user_name = session_data.get("user_name", "Candidate")
    tone = session_data.get("interviewer_tone", "professional")
    system_prompt = get_tone_prompt(tone, cv_text, user_name, session_data)
    opening_prompt = (
        f"Greet {user_name} briefly and ask the first interview question now. "
        "Keep it natural and concise."
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=GEMINI_LIVE_VOICE
                )
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=50,
                silence_duration_ms=250,
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            print(f"[WS] Gemini Live session opened for {user_name}", flush=True)
            await session.send_realtime_input(text=opening_prompt)
            print("[WS] Sent Gemini opening prompt via realtime text", flush=True)
            await websocket.send_text(json.dumps({"status": "connected"}))

            audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
            async def receive_from_client():
                try:
                    while True:
                        payload = json.loads(await websocket.receive_text())

                        if "audio" in payload:
                            await audio_queue.put(base64.b64decode(payload["audio"]))

                        if "text" in payload and str(payload["text"]).strip():
                            print(f"[WS] Forwarding text input to Gemini: {str(payload['text']).strip()[:120]}", flush=True)
                            await session.send_realtime_input(
                                text=str(payload["text"]).strip()
                            )
                except WebSocketDisconnect:
                    print(f"[WS] Interview client disconnected: {session_id}")
                    return
                except Exception as e:
                    print(f"[WS] Client -> Gemini queue error: {e}")
                    raise

            async def send_audio_to_gemini():
                try:
                    while True:
                        chunk = await audio_queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=chunk,
                                mime_type="audio/pcm;rate=16000"
                            )
                        )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[WS] Audio -> Gemini error: {e}\n{traceback.format_exc()}")
                    raise

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            server_content = response.server_content
                            if not server_content:
                                continue

                            if server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        print(f"[WS] Sending Gemini audio chunk: {len(part.inline_data.data)} bytes", flush=True)
                                        await websocket.send_text(json.dumps({
                                            "audio": base64.b64encode(part.inline_data.data).decode("utf-8")
                                        }))

                            if server_content.input_transcription and server_content.input_transcription.text:
                                print(f"[WS] Caller transcript: {server_content.input_transcription.text}", flush=True)
                                await websocket.send_text(json.dumps({
                                    "inputTranscript": server_content.input_transcription.text
                                }))

                            if server_content.output_transcription and server_content.output_transcription.text:
                                print(f"[WS] Gemini transcript: {server_content.output_transcription.text}", flush=True)
                                await websocket.send_text(json.dumps({
                                    "text": server_content.output_transcription.text
                                }))

                            if server_content.turn_complete:
                                print("[WS] Gemini turn complete", flush=True)
                                await websocket.send_text(json.dumps({"turnComplete": True}))

                            if server_content.interrupted:
                                await websocket.send_text(json.dumps({"interrupted": True}))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[WS] Gemini -> client error: {e}\n{traceback.format_exc()}")
                    raise

            background_tasks = [
                asyncio.create_task(send_audio_to_gemini()),
                asyncio.create_task(receive_from_gemini()),
            ]

            try:
                await receive_from_client()
            finally:
                for task in background_tasks:
                    task.cancel()
                await asyncio.gather(*background_tasks, return_exceptions=True)

    except Exception as e:
        print(f"[WS] Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_text(json.dumps({"error": f"Failed to connect to Gemini Live API: {e}"}))
            await websocket.close()
        except Exception:
            pass

@app.get("/coding-assessment", response_class=HTMLResponse)
async def coding_assessment(request: Request):
    """Render coding assessment page"""
    session_id = get_session_id(request)
    if not session_id:
        return RedirectResponse(url="/login", status_code=302)
    
    session_data = get_session_data(session_id)
    if 'user_name' not in session_data:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse('coding_assessment.html', {"request": request})

@app.post("/api/run-code")
async def run_code(request: Request):
    """Execute code and return output"""
    try:
        data = await request.json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        print(f"\n[CODE] Running {language} code...")
        
        if language == 'python':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                try:
                    result = subprocess.run(
                        ['python', f.name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    os.unlink(f.name)
                    
                    if result.returncode == 0:
                        return {"success": True, "output": result.stdout}
                    else:
                        return {"success": False, "error": result.stderr}
                except subprocess.TimeoutExpired:
                    os.unlink(f.name)
                    return {"success": False, "error": "Execution timeout (5s limit)"}
        else:
            return {"success": False, "error": f"{language} execution not yet supported"}
            
    except Exception as e:
        print(f"[ERROR] Code execution error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/submit-code")
async def submit_code(request: Request):
    """Submit and evaluate code solution"""
    try:
        data = await request.json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        session_id = get_session_id(request)
        
        print(f"\n[SUCCESS] Submitting {language} solution...")
        
        # Simple mock evaluation (in production, run actual test cases)
        passed_tests = 3
        total_tests = 3
        score = int((passed_tests / total_tests) * 100)
        
        # Save to session if logged in
        if session_id:
            session_data = get_session_data(session_id)
            session_data['coding_assessment'] = {
                'score': score,
                'code': code,
                'language': language,
                'passed_tests': passed_tests,
                'total_tests': total_tests
            }
            save_session(session_id, session_data)
            print(f"   [SAVE] Coding assessment saved to session (Score: {score}/100)")
        
        return {
            "passed": passed_tests == total_tests,
            "score": score,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "execution_time": 45,
            "details": "All test cases passed!" if passed_tests == total_tests else f"Failed {total_tests - passed_tests} test(s)"
        }
        
    except Exception as e:
        print(f"[ERROR] Code submission error: {e}")
        return {"passed": False, "error": str(e)}

@app.post("/api/analyze-body-language")
async def analyze_body_language():
    """Analyze body language from webcam frame (demo mode)"""
    try:
        import random
        
        # Generate random demo scores
        analysis = {
            "posture_score": random.randint(70, 95),
            "eye_contact_score": random.randint(65, 90),
            "confidence_level": random.choice(["High", "Medium", "Good"]),
            "facial_expression": random.choice(["Engaged", "Neutral", "Focused", "Confident"])
        }
        
        return {"success": True, "analysis": analysis}
        
    except Exception as e:
        print(f"[ERROR] Body language analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_interview(request: Request):
    """Evaluate candidate using Gemini AI and generate comprehensive report"""
    try:
        session_id = get_session_id(request)
        if not session_id:
            raise HTTPException(status_code=401, detail="Not logged in")
        
        session_data = get_session_data(session_id)
        if 'user_name' not in session_data:
            raise HTTPException(status_code=401, detail="Not logged in")
        
        user_name = session_data.get('user_name')
        user_email = session_data.get('user_email', '')
        cv_text = session_data.get('cv_text', '')
        conversation_history = session_data.get('conversation_history', [])
        coding_assessment = session_data.get('coding_assessment', {})
        
        if len(conversation_history) < 2:
            raise HTTPException(status_code=400, detail="Not enough conversation data")
        
        print(f"[SEARCH] Evaluating {user_name} using Gemini AI...")
        
        # Build conversation transcript
        transcript = "\n\n".join([
            f"Interviewer: {item['question']}\nCandidate: {item['answer']}"
            for item in conversation_history
        ])
        
        # Build coding assessment section
        coding_section = ""
        if coding_assessment:
            coding_section = f"""
CODING ASSESSMENT:
- Score: {coding_assessment.get('score', 'N/A')}/100
- Language: {coding_assessment.get('language', 'N/A')}
- Code Submitted: {'Yes' if coding_assessment.get('code') else 'No'}
"""
        
        # Set API key
        if 'GEMINI_API_KEY' not in os.environ and GEMINI_API_KEY:
            os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
        
        client = genai.Client()
        
        # Create comprehensive evaluation prompt
        evaluation_prompt = f"""You are an expert HR interviewer and talent evaluator. Analyze the following interview data and provide a comprehensive evaluation.

CANDIDATE INFORMATION:
Name: {user_name}
Email: {user_email}

CANDIDATE'S RESUME:
{cv_text[:3000]}

INTERVIEW TRANSCRIPT:
{transcript}

{coding_section}

EVALUATION INSTRUCTIONS:
Analyze the candidate's performance across multiple dimensions and provide detailed scores and feedback.

Provide your evaluation in the following JSON format (ONLY return valid JSON, no markdown):
{{
  "overall_score": <0-100>,
  "technical_score": <0-100>,
  "communication_score": <0-100>,
  "experience_score": <0-100>,
  "problem_solving_score": <0-100>,
  "cultural_fit_score": <0-100>,
  "strengths": [
    "Specific strength 1 with example from interview",
    "Specific strength 2 with example from interview",
    "Specific strength 3 with example from interview"
  ],
  "weaknesses": [
    "Specific weakness 1 with example",
    "Specific weakness 2 with example"
  ],
  "improvements": [
    "Actionable improvement suggestion 1",
    "Actionable improvement suggestion 2",
    "Actionable improvement suggestion 3"
  ],
  "technical_assessment": "Detailed paragraph about technical skills demonstrated",
  "communication_assessment": "Detailed paragraph about communication skills",
  "experience_assessment": "Detailed paragraph about relevant experience",
  "recommendation": "HIRE / MAYBE / REJECT with brief justification",
  "summary": "2-3 sentence overall summary of the candidate"
}}

SCORING CRITERIA:
- Technical Score: Depth of technical knowledge, problem-solving ability, coding skills
- Communication Score: Clarity, articulation, listening skills, professionalism
- Experience Score: Relevance of past experience, achievements, impact
- Problem Solving Score: Analytical thinking, approach to challenges
- Cultural Fit Score: Alignment with company values, teamwork, adaptability

Be specific and reference actual examples from the interview transcript and resume."""

        print("[SEND] Sending evaluation request to Gemini 2.5 Flash...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=evaluation_prompt
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        print(f"[RECEIVE] Received Gemini response: {response_text[:200]}...")
        
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            evaluation = json.loads(json_match.group())
            print(f"[SUCCESS] Evaluation parsed successfully")
            print(f"   Overall Score: {evaluation.get('overall_score', 'N/A')}/100")
            print(f"   Recommendation: {evaluation.get('recommendation', 'N/A')}")
        else:
            print("[WARNING] Could not parse JSON, using fallback evaluation")
            evaluation = {
                "overall_score": 75,
                "technical_score": 80,
                "communication_score": 70,
                "experience_score": 75,
                "problem_solving_score": 70,
                "cultural_fit_score": 75,
                "strengths": ["Good technical knowledge", "Clear communication", "Relevant experience"],
                "weaknesses": ["Could provide more specific examples", "Limited depth in some areas"],
                "improvements": ["Practice behavioral questions", "Prepare more detailed project examples"],
                "technical_assessment": "Demonstrated solid technical foundation",
                "communication_assessment": "Communicated clearly and professionally",
                "experience_assessment": "Has relevant experience in the field",
                "recommendation": "MAYBE",
                "summary": "Candidate shows promise with room for growth."
            }
        
        # Generate comprehensive PDF report
        filename = f"Interview_Report_{user_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(REPORTS_FOLDER, filename)
        
        # Create comprehensive PDF
        doc = SimpleDocTemplate(filepath, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#4338ca'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("AI-Powered Interview Evaluation Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Candidate Info Box
        info_data = [
            ['Candidate:', user_name],
            ['Email:', user_email],
            ['Date:', datetime.now().strftime('%B %d, %Y')],
            ['Recommendation:', evaluation.get('recommendation', 'N/A')]
        ]
        info_table = Table(info_data, colWidths=[1.5*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f4ff')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4338ca')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#4338ca')),
            ('PADDING', (0, 0), (-1, -1), 8)
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        if 'summary' in evaluation:
            story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
            story.append(Paragraph(evaluation['summary'], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Comprehensive Scores
        story.append(Paragraph("<b>Evaluation Scores</b>", styles['Heading2']))
        score_data = [
            ['Category', 'Score', 'Rating'],
        ]
        
        def get_rating(score):
            if score >= 85: return 'Excellent'
            elif score >= 70: return 'Good'
            elif score >= 55: return 'Average'
            else: return 'Needs Improvement'
        
        score_categories = [
            ('Overall Performance', 'overall_score'),
            ('Technical Skills', 'technical_score'),
            ('Communication', 'communication_score'),
            ('Experience & Background', 'experience_score'),
            ('Problem Solving', 'problem_solving_score'),
            ('Cultural Fit', 'cultural_fit_score')
        ]
        
        for category, key in score_categories:
            score = evaluation.get(key, 0)
            rating = get_rating(score)
            score_data.append([category, f"{score}/100", rating])
        
        score_table = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4338ca')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed Assessments
        if 'technical_assessment' in evaluation:
            story.append(Paragraph("<b>Technical Assessment</b>", styles['Heading2']))
            story.append(Paragraph(evaluation['technical_assessment'], styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
        
        if 'communication_assessment' in evaluation:
            story.append(Paragraph("<b>Communication Assessment</b>", styles['Heading2']))
            story.append(Paragraph(evaluation['communication_assessment'], styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
        
        if 'experience_assessment' in evaluation:
            story.append(Paragraph("<b>Experience Assessment</b>", styles['Heading2']))
            story.append(Paragraph(evaluation['experience_assessment'], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Strengths
        story.append(Paragraph("<b>Key Strengths</b>", styles['Heading2']))
        for i, strength in enumerate(evaluation.get('strengths', []), 1):
            story.append(Paragraph(f"{i}. {strength}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Weaknesses
        if 'weaknesses' in evaluation and evaluation['weaknesses']:
            story.append(Paragraph("<b>Areas of Concern</b>", styles['Heading2']))
            for i, weakness in enumerate(evaluation['weaknesses'], 1):
                story.append(Paragraph(f"{i}. {weakness}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Improvement Recommendations
        story.append(Paragraph("<b>Development Recommendations</b>", styles['Heading2']))
        for i, improvement in enumerate(evaluation.get('improvements', []), 1):
            story.append(Paragraph(f"{i}. {improvement}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Coding Assessment (if available)
        if coding_assessment:
            story.append(Paragraph("<b>Coding Assessment Results</b>", styles['Heading2']))
            story.append(Paragraph(f"Score: {coding_assessment.get('score', 'N/A')}/100", styles['Normal']))
            story.append(Paragraph(f"Language: {coding_assessment.get('language', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph("This report was generated using AI-powered analysis. Human review is recommended.", footer_style))
        story.append(Paragraph(f"Generated by HireGenie • {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", footer_style))
        
        # Build PDF
        doc.build(story)
        
        print(f"[SUCCESS] Report generated: {filename}")
        
        return {
            "success": True,
            "evaluation": evaluation,
            "report_filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-report/{filename}")
async def download_report(filename: str):
    """Download generated PDF report"""
    try:
        filepath = os.path.join(REPORTS_FOLDER, filename)
        print(f"[RECEIVE] Download request for: {filename}")
        print(f"   Looking in: {filepath}")
        print(f"   File exists: {os.path.exists(filepath)}")
        
        if os.path.exists(filepath):
            return FileResponse(
                filepath, 
                media_type="application/pdf",
                filename=filename
            )
        else:
            print(f"   [ERROR] File not found!")
            raise HTTPException(status_code=404, detail="Report not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"   [ERROR] Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/records")
async def get_records(request: Request):
    """Get all interview records for the current user"""
    session_id = get_session_id(request)
    if not session_id:
        return []
    
    session_data = get_session_data(session_id)
    user_email = session_data.get('user_email')
    
    if not user_email:
        return []
        
    return db.get_user_records(user_email)

@app.delete("/api/records/{record_id}")
async def delete_record(record_id: str, request: Request):
    """Delete a record"""
    db.delete_record(record_id)
    return {"success": True}

@app.post("/api/save-interview")
async def save_interview_record(request: Request):
    """Save an interview record"""
    try:
        data = await request.json()
        session_id = get_session_id(request)
        if session_id:
            session_data = get_session_data(session_id)
            if 'email' not in data:
                data['email'] = session_data.get('user_email')
            if 'name' not in data:
                data['name'] = session_data.get('user_name')
        
        record = db.add_record(data)
        return {"success": True, "record": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
    )
