"""Optional score gate, not a calibrated evidence-sufficiency estimator."""
from dataclasses import dataclass, field
import math

@dataclass
class CRAGVerdict:
    is_sufficient: bool | None = None
    confidence: float | None = None
    detected_gaps: list[str] = field(default_factory=list)
    suggested_rewrites: list[str] = field(default_factory=list)
    verdict: str = 'NOT_ASSESSED'
    abstain: bool = False

class RetrievalQualityGate:
    def __init__(self, threshold=None):
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError('Gate threshold must be finite')
        self.threshold=threshold
    def evaluate_retrieval(self, query, retrieved_chunks, flow_conditions=None):
        if not retrieved_chunks:
            return CRAGVerdict(False,detected_gaps=['No evidence'],verdict='NO_EVIDENCE',abstain=True)
        if self.threshold is None: return CRAGVerdict()
        scores=[]
        for chunk in retrieved_chunks:
            for key in ('rerank_score', 'score'):
                value=chunk.get(key)
                if value is None or isinstance(value, bool): continue
                try: score=float(value)
                except (TypeError, ValueError, OverflowError): continue
                if math.isfinite(score):
                    scores.append(score)
                    break
        if not scores:
            return CRAGVerdict(detected_gaps=['No valid retrieval scores'],
                               verdict='INVALID_SCORES',abstain=True)
        rejected=max(scores)<self.threshold
        return CRAGVerdict(verdict='BELOW_THRESHOLD' if rejected else 'ABOVE_THRESHOLD',abstain=rejected)
