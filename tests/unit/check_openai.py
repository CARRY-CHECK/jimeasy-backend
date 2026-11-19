# -*- coding: utf-8 -*-
"""OpenAI 호출이 가능한지 토큰 1개짜리 요청으로 확인한다. 키·요청 본문은 출력하지 않는다.

python -m tests.unit.check_openai

종료 코드 0이면 호출 가능, 1이면 불가(원인 코드 출력), 2이면 키 없음. 구조화·판정 LLM이 필요한 실행 전에 먼저 돌린다.
"""
import os
import sys

from dotenv import load_dotenv

from app.settings import ENV_FILE, MODEL_CONFIG


def main():
    load_dotenv(ENV_FILE, override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        print('OPENAI_API_KEY가 없습니다(.env 확인).')
        return 2
    from openai import OpenAI, OpenAIError, RateLimitError
    model = MODEL_CONFIG['answer']['model']
    try:
        usage = OpenAI(max_retries=0, timeout=30).chat.completions.create(
            model=model, messages=[{'role': 'user', 'content': 'ok'}], max_tokens=1).usage
    except RateLimitError as error:
        body = getattr(error, 'body', None) or {}
        detail = body.get('error', body) if isinstance(body, dict) else {}
        print(f"호출 불가: HTTP {error.status_code} {detail.get('code') or detail.get('type') or ''}".strip())
        return 1
    except OpenAIError as error:
        print(f'호출 불가: {type(error).__name__}')
        return 1
    print(f'호출 가능: {model}, 사용 토큰 {usage.total_tokens}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
