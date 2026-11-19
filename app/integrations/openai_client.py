"""OpenAI SDK 직접 호출과 LangChain 호출을 각각 제공한다."""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from openai import APIConnectionError, APIStatusError, OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import ValidationError
from app.settings import MODEL_CONFIG

_usage_log = ContextVar('carry_check_usage', default=None)


class _UsageHandler(BaseCallbackHandler):
    def __init__(self, role, log):
        self.role, self.log, self.started = role, log, None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.started = perf_counter()
        self.log.append(dict(role=self.role, status='started', tokens=None))

    def on_llm_end(self, response, **kwargs):
        message = response.generations[0][0].message
        self.log[-1].update(status='completed', tokens=getattr(message, 'usage_metadata', None),
                            seconds=round(perf_counter() - self.started, 3))

    def on_llm_error(self, error, **kwargs):
        self.log[-1].update(status='error', seconds=round(perf_counter() - self.started, 3))


@contextmanager
def collect_usage():
    log = []
    token = _usage_log.set(log)
    try:
        yield log
    finally:
        _usage_log.reset(token)


def create_chat_model(role):
    """도구 선택·판정·표현이 같은 생성 설정과 요청별 사용량 수집을 사용한다."""
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('로컬 .env에 OPENAI_API_KEY를 설정하세요.')
    config = MODEL_CONFIG[role]
    log = _usage_log.get()
    callbacks = {'callbacks': [_UsageHandler(role, log)]} if log is not None else {}
    return ChatOpenAI(api_key=key, model=config['model'], **config['parameters'],
                      timeout=config['timeout'], max_retries=config['max_retries'], **callbacks)


def generate_answer_openai(system, user):
    """설정 파일의 답변 모델에 프롬프트를 전달하고 답변 텍스트를 반환한다."""
    key = os.environ.get('OPENAI_API_KEY')
    config = MODEL_CONFIG['answer']
    model = config['model']
    if not key:
        raise ValueError('로컬 .env에 OPENAI_API_KEY를 설정하세요.')
    try:
        with OpenAI(api_key=key, timeout=config['timeout'], max_retries=config['max_retries']) as llm:
            response = llm.responses.create(
                model=model,
                instructions=system,
                input=user,
                store=False,
                **config['parameters'],
            )
    except APIStatusError as error:
        raise RuntimeError(f'OpenAI 요청 실패 (HTTP {error.status_code}). 키·모델 접근 권한·사용 한도를 확인하세요.') from None
    except APIConnectionError:
        raise RuntimeError('OpenAI 연결에 실패했습니다.') from None
    if response.status != 'completed':
        raise RuntimeError('OpenAI 답변 생성이 완료되지 않았습니다. 설정을 확인하고 다시 실행하세요.')
    text = response.output_text.strip()
    if not text:
        raise RuntimeError('OpenAI 응답에 최종 답변 텍스트가 없습니다.')
    return text


def generate_answer_langchain(system, user):
    """ChatOpenAI에 프롬프트를 전달하고 답변 텍스트를 반환한다."""
    key = os.environ.get('OPENAI_API_KEY')
    config = MODEL_CONFIG['answer']
    model = config['model']
    if not key:
        raise ValueError('로컬 .env에 OPENAI_API_KEY를 설정하세요.')
    try:
        llm = ChatOpenAI(
            api_key=key,
            model=model,
            **config['parameters'],
            timeout=config['timeout'],
            max_retries=config['max_retries'],
        )
        response = llm.invoke([('system', system), ('human', user)], store=False)
    except APIStatusError as error:
        raise RuntimeError(f'OpenAI 요청 실패 (HTTP {error.status_code}). 키·모델 접근 권한·사용 한도를 확인하세요.') from None
    except APIConnectionError:
        raise RuntimeError('OpenAI 연결에 실패했습니다.') from None
    if response.response_metadata.get('finish_reason') in {'length', 'content_filter'}:
        raise RuntimeError('OpenAI 답변 생성이 완료되지 않았습니다.')
    text = StrOutputParser().invoke(response).strip()
    if not text:
        raise RuntimeError('OpenAI 응답에 최종 답변 텍스트가 없습니다.')
    return text


def generate_structured_langchain(system, user, schema, *, role='answer'):
    """전달받은 스키마와 필드 설명을 모델에 전달하고 구조화 결과를 반환한다."""
    key = os.environ.get('OPENAI_API_KEY')
    config = MODEL_CONFIG[role]
    model = config['model']
    if not key:
        raise ValueError('로컬 .env에 OPENAI_API_KEY를 설정하세요.')
    try:
        llm = create_chat_model(role)
        structured_llm = llm.with_structured_output(schema, **config['structured_output'])
        response = structured_llm.invoke([('system', system), ('human', user)], store=False)
    except APIStatusError as error:
        raise RuntimeError(f'OpenAI 구조화 요청 실패 (HTTP {error.status_code}).') from None
    except APIConnectionError:
        raise RuntimeError('OpenAI 구조화 요청 연결에 실패했습니다.') from None
    except ValidationError:
        raise RuntimeError('OpenAI 구조화 응답이 요청한 스키마와 일치하지 않습니다.') from None
    if not isinstance(response, schema):
        raise RuntimeError('OpenAI에서 유효한 구조화 응답을 받지 못했습니다.')
    return response


# 텍스트 호출 호환용 참조. 실제 POC는 추출·판정 모두 generate_structured_langchain을 사용한다.
generate_answer = generate_answer_openai
