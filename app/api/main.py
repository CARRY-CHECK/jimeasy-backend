"""챗봇 HTTP API. POST /chat 대화, POST /chat/reset 세션 초기화.
실행: uvicorn app.api.main:app --reload  (확인: http://localhost:8000/docs)
"""
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chat import service
from app.chat.schemas import ChatRequest, ChatResponse
from app.search import warmup
from app.settings import ENV_FILE, RETRIEVAL

load_dotenv(ENV_FILE, override=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    """시작할 때 임베딩·reranker 모델을 미리 불러온다(첫 질문 지연 감소). 실패해도 서버는 뜨고 첫 질문에서 오류를 낸다."""
    if RETRIEVAL.get('warmup'):
        try:
            warmup()
        except Exception as error:
            logger.error('검색 사전 점검 실패(서버는 계속 실행): %s', str(error)[:300] if isinstance(error, RuntimeError) else type(error).__name__)
    yield


app = FastAPI(title='JimEasy', lifespan=lifespan)
app.add_middleware(  # 로컬 프론트 개발 서버 (Vite, Next.js 기본 포트)
    CORSMiddleware, allow_origins=['http://localhost:5173', 'http://localhost:3000'],
    allow_methods=['*'], allow_headers=['*'])


class ResetRequest(BaseModel):
    session_id: str


@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return service.handle(request)
    except service.ChatInputError as error:
        raise HTTPException(400, str(error)) from None
    except (RuntimeError, ValueError) as error:
        raise HTTPException(502, str(error)) from None


@app.post('/chat/reset', response_model=ChatResponse)
def reset(request: ResetRequest):
    session = service.store.reset(request.session_id)
    return ChatResponse(session_id=request.session_id, conditions=session.conditions)
