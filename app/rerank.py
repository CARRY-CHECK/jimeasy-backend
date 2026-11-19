"""cross-encoder reranker. 모델은 처음 쓸 때 한 번 불러오고 요청은 한 번에 하나씩 처리한다.

설정은 config/setting.json의 retrieval.reranker를 따른다. 모델을 불러오지 못하면 Hybrid 순서로 조용히 대체하지 않고
오류를 낸다(설정에서 rerank_pool을 0으로 두면 reranking을 끈다).
"""
import logging
import threading

from app.settings import RETRIEVAL

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_model = None


def _device(name):
    if name != 'auto':
        return name
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def get_reranker():
    """설정된 cross-encoder를 반환한다(최초 호출에서 로딩)."""
    global _model
    with _lock:
        if _model is None:
            config = RETRIEVAL['reranker']
            try:
                import torch
                from sentence_transformers import CrossEncoder
                kwargs = {'torch_dtype': torch.float16} if config.get('dtype') == 'fp16' else {}
                device = _device(config.get('device', 'auto'))
                _model = CrossEncoder(config['model'], device=device,
                                      max_length=config.get('max_length', 512), model_kwargs=kwargs)
                logger.info('reranker %s 로딩 완료, 장치 %s', config['model'], device)
                if device == 'cpu':
                    logger.warning('reranker가 CPU에서 실행됩니다. GPU가 있는데도 이 메시지가 나오면 CUDA를 지원하는 PyTorch를 설치했는지 '
                                   '확인하세요(torch.cuda.is_available()). CPU에서는 물품당 수십 초가 걸릴 수 있습니다.')
            except Exception as error:  # 모델 파일·네트워크·메모리 문제를 서비스 오류로 알린다
                raise RuntimeError(f"reranker({config['model']})를 불러오지 못했습니다: {type(error).__name__}. "
                                   '모델을 내려받았는지 확인하거나 retrieval.rerank_pool을 0으로 설정해 reranking을 끄세요.') from None
        return _model


def rerank(query, documents, top_k):
    """문서를 (쿼리, 본문) 점수로 다시 정렬해 상위 top_k개를 반환한다. 점수는 문서의 rerank_score에 기록한다.

    점수가 같으면 입력 순서를 유지한다. 원본 문서 dict는 바꾸지 않는다.
    """
    if not documents:
        return []
    model = get_reranker()
    with _lock:  # GPU 작업은 한 번에 하나만
        scores = model.predict([(query, doc['text']) for doc in documents],
                               batch_size=RETRIEVAL['reranker'].get('batch_size', 16), show_progress_bar=False)
    order = sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))
    return [dict(documents[index], rerank_score=float(scores[index])) for index in order[:top_k]]
