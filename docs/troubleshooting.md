# 설치와 문제 해결

## 환경별 설치

Python **3.12**를 쓴다. 의존성 버전은 `requirements.txt`에 고정했다.

| 환경 | 설치 | 비고 |
|---|---|---|
| Windows x64 + NVIDIA GPU | `pip install -r requirements-windows.txt` | 공통 의존성 + `torch 2.11.0+cu128`(CUDA 12.8 wheel). RTX 4070 Laptop에서 확인 |
| Apple Silicon Mac | `pip install -r requirements-mac.txt` | 공통 의존성 + `torch 2.7.1`(macOS 14에서 MPS 실행 확인). `.env`에 `CARRY_CHECK_MODEL_CONFIG=config/mac_model.json` |
| 그 밖의 환경(CPU) | `pip install -r requirements.txt` | 동작하지만 reranker가 CPU에서 돌아 느리다 |

처음 실행할 때 Hugging Face에서 `BAAI/bge-m3`(임베딩)와 `BAAI/bge-reranker-v2-m3`(재정렬)를 내려받는다. 각각 약 2.3GB다.

### Windows

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-windows.txt

# PowerShell 세션마다: 한글 출력과 자식 프로세스 인코딩을 UTF-8로 맞춘다
$env:PYTHONUTF8 = '1'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `py`가 다른 버전(예: 3.14)을 고를 수 있으므로 `-3.12`를 명시한다.
- PyTorch를 바꿔 설치할 때는 해당 가상환경의 Python 프로세스를 모두 끝내야 DLL 잠금 오류가 나지 않는다.
- 경로가 긴 파일 때문에 Git 오류가 나면 레포에서 `git config core.longpaths true`를 설정한다.
- here-string을 `python -`에 파이프로 넘길 때 `$OutputEncoding`이 UTF-8이 아니면 한글이 `?`로 바뀐다.

### Apple Silicon

```sh
python3.12 -m venv .venv && source .venv/bin/activate   # ARM64 Python인지 확인
pip install -r requirements-mac.txt
echo "CARRY_CHECK_MODEL_CONFIG=config/mac_model.json" >> .env
python -m scripts.store_chroma --batch-size 8           # MPS 메모리 때문에 작은 배치
```

Intel(x86_64) Python으로 만든 가상환경에서는 임베딩 런타임이 맞지 않을 수 있다. `python -c "import platform; print(platform.machine())"`가 `arm64`인지 확인한다.

## 실행 전 점검

```bash
python -m unittest discover -s tests/unit -p "test_*.py"   # 오프라인, 키 불필요
python -m tests.unit.smoke_service_pipeline                 # 실제 모델·DB, LLM은 가짜로 대체
python -m tests.unit.check_openai                           # OpenAI에 1토큰 요청(소액 과금)
```

`smoke_service_pipeline`은 설정 값, DB 점검, reranker 장치, Hybrid 검색과 국가·항공사 필터, 근거 ID 매핑, 대화 도구 루프, 응답 직렬화를 15개 항목으로 확인하고 지연을 출력한다.

## 자주 만나는 문제

| 증상 | 원인과 해결 |
|---|---|
| 로그에 `reranker가 CPU에서 실행됩니다` | GPU가 있다면 CUDA 빌드 PyTorch가 아니다. `torch.cuda.is_available()`이 `True`인지 확인하고 `requirements-windows.txt`로 다시 설치한다 |
| `reranker(...)를 불러오지 못했습니다` | 모델을 내려받지 못했거나 메모리가 부족하다. 네트워크를 확인하거나 `config/setting.json`의 `retrieval.rerank_pool`을 `0`으로 두어 재정렬을 끈다 |
| 서버 시작 로그에 `DB 청크 수 ...가 청크 파일 ...과 다릅니다` | 다른 데이터로 적재한 DB가 남아 있다. `paths.chroma`나 `chroma_collection`을 새 값으로 바꾸거나 `data/chroma/`를 지우고 다시 적재한다 |
| `필터용 metadata ...가 없습니다` | 청크의 `country`, `source_type`, `org_name`이 비어 있다. [data.md](data.md#청크-형식)의 형식을 확인한다 |
| `/chat`이 HTTP 502, `... 요청 실패 (HTTP 429)` | OpenAI 사용 한도나 요금 한도에 걸렸다. `python -m tests.unit.check_openai`로 확인한다 |
| `OPENAI_API_KEY를 설정하세요` | `.env`가 레포 루트에 있는지, 키 이름이 맞는지 확인한다 |
| `대화 처리 호출 상한에 도달했습니다` | 대화 진행 모델이 8번 안에 끝내지 못했다. 조건을 구체적으로 적어 다시 묻거나 `config/model.json`의 `chat.max_steps`를 조정한다 |
| 프론트엔드에서 CORS 오류 | 허용 출처는 `http://localhost:5173`, `http://localhost:3000`뿐이다. `app/api/main.py`의 `allow_origins`를 바꾼다 |
| 평가 도구 실행 중 메모리 부족 | 전체 말뭉치로 검색 평가를 돌리면 프로세스 메모리가 약 10GB까지 오른다. 다른 무거운 작업과 함께 돌리지 않는다 |

## 플랫폼 간 차이

검색 로직과 `config/setting.json`은 운영체제와 무관하다. 달라지는 것은 장치뿐이다.

| 항목 | Mac | Windows / Linux |
|---|---|---|
| 임베딩 장치 | `config/mac_model.json`(MPS) | `config/model.json`(CPU) |
| reranker 장치 | `auto` → MPS | `auto` → CUDA, 없으면 CPU |
| Chroma DB | 장치마다 따로 적재 | 장치마다 따로 적재 |

같은 BGE-M3라도 장치가 다르면 Dense 순위가 아주 조금 달라질 수 있다(원래 말뭉치에서 상위 5개 순서가 검색 77건 중 75건 일치). DB 파일을 다른 장치에서 복사해 쓰기보다 그 장치에서 다시 적재한다.
CRLF와 LF는 내용이 같아도 파일 해시가 다르므로, 플랫폼 간에 실행 기록의 해시를 비교할 때는 내용도 함께 확인한다.
