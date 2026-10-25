"""Scientific RAG configuration and tokenizer identity."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RAGConfig:
    dense_model: str = 'BAAI/bge-small-en-v1.5'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    dense_revision: str | None = None
    reranker_revision: str | None = None
    retrieval_k: int = 30
    rerank_k: int = 30
    evidence_k: int = 5
    context_budget: int = 2048
    gate_threshold: float | None = None
    expand_parents: bool = True
    query_instruction: str = 'Represent this sentence for searching relevant passages: '
    def __post_init__(self):
        if min(self.retrieval_k,self.rerank_k,self.evidence_k,self.context_budget)<=0:
            raise ValueError('All sizes must be positive')
        if self.rerank_k>self.retrieval_k: raise ValueError('rerank_k must not exceed retrieval_k')
        if self.evidence_k>self.rerank_k: raise ValueError('evidence_k must not exceed rerank_k')
    def to_dict(self): return asdict(self)

def load_token_counter(name):
    if name.startswith('hf:'):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(name[3:])
        # Counting context is not an encoder forward pass and must not truncate it.
        tok.model_max_length=10**12
        return lambda text:len(tok.encode(text,add_special_tokens=False))
    if name.startswith('tiktoken:'):
        import tiktoken
        enc=tiktoken.get_encoding(name.split(':',1)[1])
        return lambda text:len(enc.encode(text,disallowed_special=()))
    raise ValueError('Specify hf:<reader tokenizer> or tiktoken:<encoding> (explicit proxy)')
