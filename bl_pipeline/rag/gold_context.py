"""Annotation-assisted reader diagnostic; never a deployable retrieval method."""
from .stage_wise_benchmark import context_metrics


def reader_diagnostic_eligible(refs):
    # Avoid automatic refusal credit and ambiguous answerability labels.
    return all(r['type'] != 'none' for r in refs) and any(r['evidence'] for r in refs)


def gold_context_trace(refs, budget, counter):
    """Select the shortest annotated evidence set; preserve whole units.

    Selection uses evidence lengths only, never generated-answer scores. If no
    complete set fits, pack units from the shortest set and expose omissions.
    Annotation coverage does not establish semantic answer sufficiency.
    """
    candidates = []
    for index, ref in enumerate(refs):
        units = list(dict.fromkeys(e for e in ref['evidence']
                                   if e.removeprefix('FLOAT SELECTED: ').strip()))
        if units:
            text = '\n\n'.join(e.removeprefix('FLOAT SELECTED: ').strip() for e in units)
            candidates.append((counter(text), index, units))
    _, reference_index, units = min(candidates) if candidates else (0, None, [])
    included, omitted, spans = [], [], []
    for index, evidence in enumerate(units):
        span = evidence.removeprefix('FLOAT SELECTED: ').strip()
        if counter('\n\n'.join(spans + [span])) <= budget:
            spans.append(span)
            included.append(index)
        else:
            omitted.append(index)
    context = '\n\n'.join(spans)
    delivered = [e for e in units if e.removeprefix('FLOAT SELECTED: ').strip() in context]
    return {
        'rankings': {k: [] for k in ('dense', 'bm25', 'hybrid', 'hybrid_rerank_pool', 'reranked')},
        'selected_evidence': units, 'delivered_evidence': delivered,
        'selected_ids': [f'gold:{reference_index}:{i}' for i in included],
        'context': context, 'context_tokens': counter(context),
        'omitted_ids': [f'gold:{reference_index}:{i}' for i in omitted],
        'gate': {'abstain': False, 'is_sufficient': None, 'verdict': 'GOLD_DIAGNOSTIC'},
        'gold_diagnostic': {
            'policy': 'shortest_nonempty_reference_whole_units_v1',
            'reference_index': reference_index,
            'eligible_for_reader_evaluation': reader_diagnostic_eligible(refs),
            'selected_reference_complete': bool(units) and set(units) <= set(delivered),
            'coverage': context_metrics(context, refs),
        },
    }
