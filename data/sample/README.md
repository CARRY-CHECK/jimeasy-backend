# 합성 샘플 데이터

> **이 폴더의 규정 문구는 실제 규정이 아닙니다.** 파이프라인을 실행해 보기 위해 팀이 직접 쓴 짧은 예시이며,
> 출처 기관·항공사는 모두 가상입니다. 실제 여행에서는 항공사·공항·정부의 공식 안내를 확인하세요.

| 파일 | 내용 | 쓰는 곳 |
|---|---|---|
| `chunks.jsonl` | 가상 출처 9개 문서, 29개 청크(한국·일본·미국·영국·중국). 실제 말뭉치와 같은 `id`·`text`·`metadata` 형식 | `config/setting.json`의 `chunk_sources`, `scripts.store_chroma` |
| `questions.json` | 5문항의 질문과 정답(물품별 기내·위탁 판정, 설명, 정답 근거 청크 ID) | `scripts.run_dataset`, `scripts.evaluate_dataset` |
| `queries.json` | 같은 5문항의 질의 구조화 정답(`expected_parsed`) | `scripts.evaluate_dataset --query-dataset` |
| `chat_scenarios.json` | 대화 흐름 시나리오 3개(조건 계승, 재질문 후 답변, 미지원 국가) | `scripts.run_chat_dataset` |

- 문구는 널리 알려진 일반 수하물 규칙(액체 100ml, 리튬 배터리 Wh 구간 등)을 짧게 바꿔 쓴 것입니다. 어느 기관의 문서도 복사하지 않았습니다.
- 모든 출처 이름에 `(합성 예시)` 또는 `(synthetic)`, 기관 이름에 `(가상)` 또는 `(fictional)`이 붙어 있고 URL은 `example.com`입니다.
- 실제 항공사 이름이 붙은 문서가 없으므로 항공사 필터(`retrieval.airline_filter`)는 샘플에서 항공사 문서를 거르는 동작만 보여 줍니다.
- 정답 라벨도 이 샘플에 맞춰 만든 것이라 점수는 실행 확인용이며 성능 지표가 아닙니다. 실제 평가 결과는 [docs/evaluation.md](../../docs/evaluation.md)를 보세요.

샘플은 원래 말뭉치(1,690청크)보다 훨씬 작습니다. 물품 하나에 최종 8개 문서를 넘기므로 관련 없는 문서도 함께 검색됩니다.
자신의 데이터로 바꾸는 방법은 [docs/data.md](../../docs/data.md#자신의-데이터로-실행하기)에 있습니다.
