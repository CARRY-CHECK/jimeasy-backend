"""config/setting.json을 읽고 경로와 시간대를 Python 객체로 변환한다.

root는 JSON 파일 위치 기준, paths의 각 경로는 root 기준이다.
설정 변경은 다음 프로세스 실행부터 적용된다.
"""
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import dotenv_values

_CONFIG_FILE = Path(__file__).resolve().parents[1] / 'config' / 'setting.json'
with _CONFIG_FILE.open(encoding='utf-8') as file:
    _settings = json.load(file)

ROOT = (_CONFIG_FILE.parent / _settings['root']).resolve()
_paths = _settings['paths']
DATA = ROOT / _paths['data']
PROMPTS_DIR = ROOT / _paths['prompts']
ENV_FILE = ROOT / _paths['env_file']
RESULTS_DIR = ROOT / _paths['results']
CHROMA_PATH = Path(_paths['chroma'])
CHROMA_COLLECTION = _settings['chroma_collection']
CHUNK_SOURCES = {country: ROOT / path for country, path in _settings['chunk_sources'].items()}
KST = ZoneInfo(_settings['timezone'])
# 물품 1개당 판정에 넘기는 최종 문서 수. 검색 방식 설정은 RETRIEVAL을 따른다.
DEFAULT_K = _settings['default_k']
RETRIEVAL = _settings['retrieval']
# 구조화 시스템 프롬프트 기본 선택(prompts/ 안의 파일명). 실험 실행은 --parse-system-prompt로 따로 지정한다.
PARSE_SYSTEM_PROMPT = _settings['parse_system_prompt']
# 대화 도구 모델별 시스템 프롬프트 선택(prompts/ 기준 경로). 판정은 단발성 RAG·대화가 함께 쓴다.
AGENT_SYSTEM_PROMPT = _settings['agent_system_prompt']
PRESENT_SYSTEM_PROMPT = _settings['present_system_prompt']
JUDGE_SYSTEM_PROMPT = _settings['judge_system_prompt']
FILTER_FIELDS = frozenset(_settings['filter_fields'])
OVERRIDE_KEYS = tuple(_settings['override_keys'])

# Only read the local config selector here; API keys are loaded by entrypoints.
_local_model_config = dotenv_values(ENV_FILE, interpolate=False).get('CARRY_CHECK_MODEL_CONFIG')
_model_config = os.environ.get('CARRY_CHECK_MODEL_CONFIG', _local_model_config)
MODEL_CONFIG_FILE = Path(_model_config).expanduser() if _model_config else _CONFIG_FILE.parent / 'model.json'
if not MODEL_CONFIG_FILE.is_absolute():
    MODEL_CONFIG_FILE = ROOT / MODEL_CONFIG_FILE
MODEL_CONFIG_FILE = MODEL_CONFIG_FILE.resolve()

# A selected file must exist; never silently fall back to a different model.
with MODEL_CONFIG_FILE.open(encoding='utf-8') as file:
    MODEL_CONFIG = json.load(file)


def _validate_retrieval(config, default_k):
    """검색 설정 오류를 서비스 시작 때 바로 드러낸다."""
    if config.get('method') not in ('dense', 'hybrid'):
        raise ValueError("retrieval.method는 'dense' 또는 'hybrid'여야 합니다.")
    for key in ('candidate_k',):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f'retrieval.{key}는 1 이상의 정수여야 합니다.')
    pool = config.get('rerank_pool')
    if pool is not None and (not isinstance(pool, int) or pool < 0):
        raise ValueError('retrieval.rerank_pool은 0 이상의 정수 또는 null이어야 합니다.')
    if pool and pool > config['candidate_k']:
        raise ValueError('retrieval.rerank_pool은 방식별 후보 수(candidate_k)를 넘을 수 없습니다.')
    if not isinstance(default_k, int) or default_k < 1:
        raise ValueError('default_k는 1 이상의 정수여야 합니다.')
    reranker = config.get('reranker') or {}
    if config['method'] == 'hybrid' and pool and not reranker.get('model'):
        raise ValueError('reranking을 쓰려면 retrieval.reranker.model이 필요합니다.')
    if reranker.get('dtype', 'fp32') not in ('fp32', 'fp16'):
        raise ValueError("retrieval.reranker.dtype은 'fp32' 또는 'fp16'이어야 합니다.")


_validate_retrieval(RETRIEVAL, DEFAULT_K)
