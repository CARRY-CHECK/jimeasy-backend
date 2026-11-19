# HTTP API

FastAPI 서버(`app/api/main.py`)는 대화 엔드포인트 두 개를 제공한다. 실행 중에는 http://127.0.0.1:8000/docs 에서 OpenAPI 문서를 볼 수 있다.

```bash
python -m uvicorn app.api.main:app --port 8000
```

시작할 때 임베딩·reranker 모델을 미리 불러오므로 준비까지 수십 초가 걸린다. 프론트엔드는 `/openapi.json` 응답으로 준비 여부를 확인할 수 있다.
CORS는 로컬 프론트엔드 개발 서버(`http://localhost:5173`, `http://localhost:3000`)만 허용한다. 다른 출처에서 호출하려면 `app/api/main.py`의 `allow_origins`를 바꾼다.

## POST /chat

텍스트 질문(`message`)이나 직전 추가 질문의 선택지(`choice`) 중 **하나만** 보낸다. 첫 요청은 `session_id` 없이 보내고, 응답의 `session_id`를 다음 요청에 넣는다.

```json
{"session_id": null, "message": "인천에서 도쿄 가는데 보조배터리 2개 기내에 가져가도 돼요?"}
```

```json
{"session_id": "7f3c...", "choice": {"question_id": "c1a2...", "card": 0, "option": 1}}
```

| 필드 | 뜻 |
|---|---|
| `choice.question_id` | 직전 응답의 `clarification.id`. 지난 질문이나 초기화 전의 값이면 400 |
| `choice.card` | `clarification.questions` 배열의 순번 |
| `choice.option` | 그 질문의 `options` 배열 순번 |

여러 추가 질문에 한 번에 답할 때는 `message`에 문장으로 적어 보내도 된다.

### 응답 (`ChatResponse`)

| 필드 | 뜻 |
|---|---|
| `session_id` | 세션 ID |
| `conditions` | 현재 여행 조건. `departure`·`arrival`(각 `place`, `airline`), `travel_date`, `baggage_type`(`기내`/`위탁`/`둘 다`/null) |
| `summary` | 결론 한 문장 |
| `message` | 판정 대신 안내가 필요할 때의 문장(예: 물품이 없는 질문, 미지원 국가) |
| `cards[]` | 물품별 판정 카드 |
| `cards[].id`, `name` | 세션 안에서 유지되는 카드 번호, 물품명 |
| `cards[].carry_on`, `cards[].checked` | 기내·위탁 판정. `status`는 `가능`/`불가`/`조건부 가능`/`확인 필요`/null, `condition`은 짧은 조건 문구 |
| `cards[].notes` | 판정의 조건·이유 설명 |
| `cards[].evidence[]` | 모델이 판정에 사용했다고 보고한 근거 문서의 metadata와 `chunk_id`(출처명, 기관, URL, 버전 등) |
| `suggested_next` | 이어서 물어볼 만한 질문 |
| `clarification` | 추가 질문이 필요할 때만 있다. `id`와 `questions[]`(`id`, `field`, `question`, `options`) 최대 3개. 선택지 마지막에 `잘 모르겠어요`가 붙는다 |
| `turn_id` | 이번 턴 ID |
| `diagnostics` | 진단 정보. 엔진, 상태·검색 버전, 대화 모델 호출 수, 도구 호출 기록, 모델별 토큰·시간(`model_calls`), 구조화 결과(`parsed_query`), 처리 시간 |

추가 질문이 있는 응답에는 판정 카드가 없을 수 있다. 화면은 `clarification.questions`를 먼저 보여 주고 사용자의 답을 다음 요청으로 보낸다.

형식 예시(값은 설명용이며 실제 실행 결과가 아니다):

```json
{
  "session_id": "7f3c...",
  "conditions": {"departure": {"place": "인천", "airline": null}, "arrival": {"place": "도쿄", "airline": null},
                 "travel_date": null, "baggage_type": "기내"},
  "summary": "100Wh 이하 보조배터리는 기내에만 가져갈 수 있습니다.",
  "message": null,
  "cards": [{
    "id": 1, "name": "보조배터리",
    "carry_on": {"status": "조건부 가능", "condition": "100Wh 이하, 단자 절연"},
    "checked": {"status": "불가", "condition": null},
    "notes": "보조배터리는 위탁 수하물로 보낼 수 없고 기내에 휴대해야 합니다. ...",
    "follow_up": null,
    "evidence": [{"chunk_id": "kr.sample_dg.0001#0", "source_name": "...", "org_name": "...", "source_url": "https://...", "...": "..."}]
  }],
  "suggested_next": "전자담배도 기내에 가져갈 수 있나요?",
  "clarification": null,
  "turn_id": "b91e...",
  "diagnostics": {"engine": "tools", "agent_calls": 3, "model_calls": ["..."], "tool_events": ["..."], "seconds": 9.8}
}
```

### 오류

| 상태 | 언제 |
|---|---|
| 400 | 빈 질문, 지난 선택지 등 입력 오류 |
| 422 | `message`와 `choice`를 둘 다 보내거나 둘 다 빠뜨린 경우 등 요청 형식 오류 |
| 502 | OpenAI 호출 실패(키·한도·연결), 모델 출력 검증 실패, 요청당 호출 상한 도달. `detail`에 사유가 있다 |

오류가 난 요청은 세션 상태를 바꾸지 않으므로 같은 질문을 다시 보낼 수 있다.

## POST /chat/reset

```json
{"session_id": "7f3c..."}
```

세션의 조건·물품·대기 중인 질문을 지우고 빈 `ChatResponse`를 돌려준다. 초기화 뒤에는 텍스트 질문부터 다시 시작한다.

## CLI

HTTP 없이 같은 흐름을 쓰는 명령도 있다. 모두 `.env`의 `OPENAI_API_KEY`가 필요하다.

| 명령 | 내용 |
|---|---|
| `python -m scripts.chat` | 터미널 대화(같은 도구 루프) |
| `python -m scripts.ask "질문" [--show-parsed]` | 단발성 질문. 결과 JSON을 `outputs/results/`에 저장한다 |
| `python -m scripts.run_dataset` | 질문 데이터셋 일괄 실행(기본 `data/sample/questions.json`, 결과 `outputs/runs/attempt_N/`) |
| `python -m scripts.run_chat_dataset` | 대화 시나리오 실행과 상태·흐름 점검(기본 `data/sample/chat_scenarios.json`) |

`scripts.ask`는 `--departure`, `--arrival`, `--departure-airline`, `--arrival-airline`, `--airline`, `--travel-date`, `--baggage-type`으로 구조화 결과를 덮어쓰고,
`--filter KEY=VALUE`(반복 가능, `country`·`source_name`·`source_type`·`version`·`effective_date`·`unit`)로 metadata 정확 일치 필터를 더할 수 있다.
`--k`는 물품당 최종 문서 수, `--retrieval dense`는 이전 검색 방식, `--no-save`는 저장하지 않고 출력만 한다.

단발성 결과 JSON에는 최종 답변(`answer.items[]`: `name`, `carry_on`, `checked_baggage`, `notes`, `source`) 외에 구조화 결과(`parsed`),
물품별 검색어와 문서(`groups`), 검색된 전체 출처(`searched_document_sources`)와 모델이 사용했다고 보고한 출처(`model_used_sources`), 모델 판정 원본(`model_decisions`),
적용한 검색 설정(`retrieval`)이 함께 저장된다.
