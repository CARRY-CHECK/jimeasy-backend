# -*- coding: utf-8 -*-
"""reranking 평가·지연 측정이 공유하는 입력 로더.

검색 평가(app.evaluate_hybrid_search) 결과 폴더의 질문별 JSON에서 물품별 후보 pool을 복원한다.
검색을 다시 실행하지 않는다. pool은 Hybrid(RRF)가 만든 전체 융합 순서(Dense 50 ∪ BM25 50, 최대 100개)이고
본문은 같은 파일의 dense_candidates·bm25_candidates에 있다.
"""
import json
from pathlib import Path

from app.rag import build_retrieval_query

ROOT = Path(__file__).resolve().parents[2]  # scripts/evaluation/ 기준 저장소 루트


def load_pools(run_dirs, records_root=ROOT / 'data/test/test9_gold_query_records'):
    """물품 하나(정답 청크가 있는 것)마다 하나의 검색 단위를 반환한다."""
    units = []
    for run in run_dirs:
        run = Path(run)
        dataset = run.name
        for path in sorted(run.glob('*.json')):
            if path.name == 'summary.json' or path.name.startswith('db_'):
                continue
            result = json.loads(path.read_text(encoding='utf-8'))
            record = json.loads((Path(records_root) / dataset / path.name).read_text(encoding='utf-8'))
            item_by_query = {g['query']: g['item'] for g in record['groups']}
            searches = {s['query']: s for s in result['searches']}
            for item in result['items']:
                if item['status'] != 'scored':
                    continue
                search = searches[item['query']]
                texts = {row['id']: row['text'] for row in search['dense_candidates'] + search['bm25_candidates']}
                traces = sorted(search['traces'], key=lambda t: t['hybrid_rank'])
                pool = [{'id': t['chunk_id'], 'text': texts[t['chunk_id']], 'hybrid_rank': t['hybrid_rank'],
                         'dense_rank': t['dense_rank'], 'bm25_rank': t['bm25_rank']} for t in traces]
                source = item_by_query[item['query']]
                units.append({'set': dataset, 'qid': result['id'], 'item': item['name'], 'query': item['query'],
                              'item_query': build_retrieval_query(source, None), 'gold': item['gold_chunk_ids'],
                              'pool': pool})
    return units


# ---------------------------------------------------------------------------
# 이어하기용 점수 저장소와 실행 잠금 (R1)
# ---------------------------------------------------------------------------
import contextlib
import os


def unit_key(unit):
    return f"{unit['set']}|{unit['qid']}|{unit['item']}"


class ScoreStore:
    """물품 단위로 reranker 점수를 JSONL에 덧붙여 저장한다. 중단돼도 끝난 물품은 남고, 다시 실행하면 건너뛴다.

    한 줄은 한 물품의 순위 구간 한 조각이다: {'key', 'rank_from', 'rank_to', 'scores'} (순위는 1부터, 양 끝 포함).
    같은 물품의 조각을 순서대로 이어 붙여 앞에서부터 빈틈없이 이어진 점수만 사용한다.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._chunks = {}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 중단으로 잘린 마지막 줄은 버린다
                self._chunks.setdefault(row['key'], []).append(row)

    def scores(self, key):
        """앞에서부터 이어진 점수 목록. 빈틈이 생기면 거기서 멈춘다."""
        out = []
        for row in sorted(self._chunks.get(key, []), key=lambda r: r['rank_from']):
            if row['rank_from'] != len(out) + 1:
                break
            out.extend(row['scores'])
        return out

    def scored_until(self, key):
        return len(self.scores(key))

    def append(self, key, rank_from, scores, **meta):
        row = {'key': key, 'rank_from': rank_from, 'rank_to': rank_from + len(scores) - 1,
               'scores': [float(s) for s in scores], **meta}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
            file.flush()
            os.fsync(file.fileno())
        self._chunks.setdefault(key, []).append(row)


class RunLocked(RuntimeError):
    pass


def _alive(pid):
    if os.name == 'nt':
        return True  # Windows에서는 os.kill(pid, 0)이 프로세스를 종료시킬 수 있어 확인하지 않는다
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def gpu_lock(path=ROOT / '.rerank_gpu.lock'):
    """GPU(MPS/CUDA) 작업이 겹쳐 느려지는 것을 막는다. 이미 실행 중이면 RunLocked를 던진다."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(path.read_text(encoding='utf-8').split()[0])
        except (ValueError, IndexError, OSError):
            pid = None
        if pid is not None and os.name != 'nt' and not _alive(pid):
            path.unlink()  # 죽은 프로세스의 잠금
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RunLocked(f'다른 reranking GPU 작업(pid {pid})이 실행 중입니다. 끝난 뒤 다시 실행하세요. '
                            f'프로세스가 없다면 {path}를 지우세요.') from None
    try:
        os.write(fd, f'{os.getpid()}\n'.encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
