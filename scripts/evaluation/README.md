# 평가·분석 도구

검색 실험(깊이·Hybrid·필터·쿼리 방식), 재정렬 채점, 서비스 검색 경로 평가, 보고서 생성에 쓴 도구다.
모두 `python -m scripts.evaluation.<이름> --help`로 옵션을 볼 수 있고, 표시한 것 외에는 LLM·OpenAI를 호출하지 않는다.

> 각 파일 맨 위 docstring의 실행 예시는 원래 비공개 평가셋과 결과 폴더(`data/test/…`, `tests/verN_test/…`)를 가리킨다.
> 공개 레포에는 해당 파일이 없으므로 경로를 `data/sample/…`, `outputs/…`로 바꿔 실행한다.

| 도구 | 하는 일 | 입력 |
|---|---|---|
| `build_gold_query_records` | 구조화 정답으로 물품별 검색 쿼리를 만들어 records 형식으로 저장(실제 구조화 오류를 뺀 검색 평가용) | 질문·구조화 정답 |
| `evaluate_service_retrieval` | 현재 서비스 검색 경로를 실제로 호출해 정답 근거 회수율을 재고 보고서를 만든다(`run`, `report`) | 질문, records, Chroma, 모델 |
| `score_rerank` | 저장된 후보의 순위 구간을 cross-encoder로 채점해 점수 파일에 덧붙인다(끊겨도 이어서 실행) | 검색 평가 결과, reranker |
| `evaluate_rerank` | 저장된 점수로 후보 크기별 재정렬 Recall@M·MRR·nDCG@10을 비교(모델 로딩 없음) | 점수 파일 |
| `benchmark_reranker`, `measure_demo_latency` | 재정렬과 물품 2개 질문의 검색 지연 측정 | records, 모델 |
| `compare_query_variants` | 쿼리 방식만 다른 두 검색 평가의 정답 변화를 자료 유형·조건별로 비교 | 검색 평가 결과 |
| `dataset_statistics` | 평가셋 정답 통계(분류·물품·라벨·근거 분포) | 질문 |
| `merge_eval_inputs` | 여러 평가셋·실행 기록을 하나의 평가 입력으로 합친다 | 질문, 구조화 정답, 실행 기록 |
| `make_present`, `make_retrieval_present` | 평가 결과(report.json 등)에서 발표용 요약 `present.md`와 CSV를 만든다 | 평가 결과 |
| `rerank_common` | 위 도구가 함께 쓰는 입력 로더 | — |

같은 계열의 도구가 다른 위치에도 있다.

| 도구 | 하는 일 |
|---|---|
| `scripts/evaluate_dataset.py` | 저장된 실행 기록의 구조화·검색·판정·설명 채점([docs/evaluation.md](../../docs/evaluation.md)) |
| `scripts/query_evaluation.py` | 구조화 정답과 저장된 parsed 비교(`evaluate_dataset`이 사용) |
| `app/evaluate_hybrid_search.py` | 저장된 쿼리로 Dense / BM25 / Hybrid 후보를 같은 조건에서 비교(`--airline-filter`, `--query-mode`, `--no-metadata-filter`) |
| `app/query_gold_similarity.py` | 검색 쿼리와 정답 청크의 코사인 유사도 진단 |
| `app/embedding_token_stats.py` | 청크의 임베딩 입력 토큰 수 통계 |

합성 샘플로 검색 실험을 돌려 보는 예:

```bash
python -m scripts.evaluation.build_gold_query_records --questions data/sample/questions.json \
    --queries data/sample/queries.json --output outputs/gold_query_records
python -m app.evaluate_hybrid_search --dataset data/sample/questions.json \
    --records outputs/gold_query_records --airline-filter
```
