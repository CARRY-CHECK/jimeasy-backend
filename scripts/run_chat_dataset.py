"""대화 시나리오 실행과 상태/흐름 검증. 규정의 의미 정확도 자동 채점은 아니다.

실행 시 실제 모델·Chroma를 사용한다. 출력에는 API 키나 .env를 기록하지 않는다.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from app.chat.schemas import ChatRequest
from app.chat.service import handle_tools
from app.chat.sessions import SessionStore
from app.settings import (AGENT_SYSTEM_PROMPT, ENV_FILE, JUDGE_SYSTEM_PROMPT, KST, MODEL_CONFIG, RESULTS_DIR,
                          PRESENT_SYSTEM_PROMPT, ROOT)


def check_turn(response, session, expected):
    errors = []
    if 'clarification' in expected and bool(response.clarification) != expected['clarification']:
        errors.append('clarification 불일치')
    if 'items' in expected and [card.name for card in response.cards] != expected['items']:
        errors.append('응답 대상 물품 불일치')
    if 'arrival' in expected and response.conditions.arrival.place != expected['arrival']:
        errors.append('도착지 계승 불일치')
    if 'details_contains' in expected and not any(expected['details_contains'] in (item['details'] or '')
                                                   for item in session.items):
        errors.append('물품 속성 수정 불일치')
    if expected.get('no_evidence') and any(card.evidence for card in response.cards):
        errors.append('미지원 구간에 근거 제공')
    return errors


def run(scenarios, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(started_at=datetime.now(KST).isoformat(), scenarios=[], provenance=dict(
        models=MODEL_CONFIG, scenarios_sha256=hashlib.sha256(json.dumps(
            scenarios, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest(),
        files_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'app/chat/service.py', 'app/chat/tools.py', 'app/chat/sessions.py', 'app/chat/schemas.py',
            'app/judgment.py', 'app/retrieval.py', 'prompts/' + AGENT_SYSTEM_PROMPT,
            'prompts/' + JUDGE_SYSTEM_PROMPT, 'prompts/' + PRESENT_SYSTEM_PROMPT)}))
    for scenario in scenarios:
        sessions, session_id = SessionStore(), None
        record = dict(id=scenario['id'], turns=[])
        for turn in scenario['turns']:
            start = perf_counter()
            try:
                response = handle_tools(ChatRequest(session_id=session_id, message=turn['message']), sessions)
                session_id = response.session_id
                session = sessions.get(session_id)[1]
                errors = check_turn(response, session, turn.get('expect', {}))
                row = dict(message=turn['message'], response=response.model_dump(),
                           state=session.snapshot(), groups=session.groups,
                           checks=errors, status='failed' if errors else 'passed')
            except (ValueError, RuntimeError) as error:
                # SDK 예외/환경 내용은 저장하지 않는다.
                row = dict(message=turn['message'], status='error', error_type=type(error).__name__)
            row['seconds'] = round(perf_counter() - start, 3)
            record['turns'].append(row)
            print(f"{scenario['id']} turn {len(record['turns'])}: {row['status']}", flush=True)
            if row['status'] == 'error':
                break
        report['scenarios'].append(record)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [turn for record in report['scenarios'] for turn in record['turns']]
    report['summary'] = dict(turns=len(rows), passed=sum(row['status'] == 'passed' for row in rows),
                             failed=sum(row['status'] != 'passed' for row in rows))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/sample/chat_scenarios.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scenario', action='append', help='실행할 시나리오 ID. 반복 지정 가능')
    args = parser.parse_args()
    load_dotenv(ENV_FILE, override=False)
    scenarios = json.loads(args.dataset.read_text(encoding='utf-8'))
    if args.scenario:
        scenarios = [row for row in scenarios if row['id'] in args.scenario]
        if len(scenarios) != len(set(args.scenario)):
            parser.error('시나리오 ID를 확인하세요.')
    output = args.output or RESULTS_DIR / ('chat_' + datetime.now(KST).strftime('%Y%m%d_%H%M%S_%f'))
    report = run(scenarios, output)
    print(json.dumps(report['summary'], ensure_ascii=False))
    print(output)
    return int(bool(report['summary']['failed']))


if __name__ == '__main__':
    raise SystemExit(main())
