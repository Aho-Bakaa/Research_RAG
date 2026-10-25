"""Benchmark metrics with explicit denominators and official QASPER semantics."""
from collections import Counter
import re
import string


def normalize_answer(text):
    text=text.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def compute_token_f1(pred,gold):
    p=normalize_answer(pred).split(); g=normalize_answer(gold).split()
    common=sum((Counter(p)&Counter(g)).values())
    return 2*common/(len(p)+len(g)) if common else 0.0


def compute_paragraph_f1(pred,gold):
    if not pred and not gold: return 1.0,1.0,1.0
    common=len(set(pred)&set(gold))
    if not common: return 0.0,0.0,0.0
    p=common/len(pred); r=common/len(gold)
    return 2*p*r/(p+r),p,r


def references(qa):
    result=[]
    for wrapper in qa['answers']:
        a=wrapper['answer']
        if a['unanswerable']: answer,kind,evidence='Unanswerable','none',[]
        else:
            evidence=a['evidence']
            if a['extractive_spans']: answer,kind=', '.join(a['extractive_spans']),'extractive'
            elif a['free_form_answer']: answer,kind=a['free_form_answer'],'abstractive'
            elif a['yes_no'] is not None: answer,kind='Yes' if a['yes_no'] else 'No','boolean'
            else: raise ValueError('Invalid QASPER annotation')
        result.append({'answer':answer,'type':kind,'evidence':evidence})
    if not result: raise ValueError('Question has no references')
    return result


def score_answer(pred,refs):
    # Stable tie order matches the official evaluator; categories can depend on prediction.
    return max(((compute_token_f1(pred,r['answer']),r['type']) for r in refs),key=lambda x:x[0])


def ranking_metrics(ranking,refs,k):
    gold=set(e for r in refs for e in r['evidence'])
    if not gold: return None
    top=ranking[:k]; found=set(e for x in top for e in x['evidence'])
    rr=next((1/(i+1) for i,x in enumerate(top) if gold.intersection(x['evidence'])),0.0)
    reference_sets=[set(r['evidence']) for r in refs if r['evidence']]
    return {'hit':float(bool(found&gold)),'union_evidence_recall':len(found&gold)/len(gold),
            'complete_reference_coverage':float(any(s<=found for s in reference_sets)),'mrr':rr}


def context_metrics(context,refs):
    sets=[r['evidence'] for r in refs if r['evidence']]
    gold=set(e for s in sets for e in s)
    spans=[]; covered=set()
    for evidence in gold:
        needle=evidence.removeprefix('FLOAT SELECTED: ').strip()
        start=context.find(needle) if needle else -1
        if start>=0: covered.add(evidence); spans.append((start,start+len(needle)))
    merged=[]
    for a,b in sorted(spans):
        if merged and a<=merged[-1][1]: merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else: merged.append((a,b))
    return {'gold_evidence_recall':len(covered)/len(gold) if gold else None,
            'complete_reference_coverage':float(any(set(s)<=covered for s in sets)) if sets else None,
            'contained_gold_character_fraction':sum(b-a for a,b in merged)/len(context) if context else 0.0}


def mean(values):
    values=[x for x in values if x is not None]
    return sum(values)/len(values) if values else None


def summarize(records,gen_ids):
    evidence=[]; answers=[]; bytype={k:[] for k in ('extractive','abstractive','boolean','none')}
    gates=Counter(); refusals=Counter(); statuses=Counter()
    for r in records:
        refs=r['references']
        evidence.append(max(compute_paragraph_f1(r['predicted_evidence'],g['evidence'])[0] for g in refs))
        kinds={x['type'] for x in refs}
        label='unanswerable' if kinds=={'none'} else 'mixed' if 'none' in kinds else 'answerable'
        gates[(label,'abstain' if r['retrieval']['gate']['abstain'] else 'pass')]+=1
        if r['question_id'] in gen_ids:
            gen=r['generation']; statuses[gen['status']]+=1
            score,kind=score_answer(r['predicted_answer'],refs); answers.append(score); bytype[kind].append(score)
            refused=gen['status'] in ('ok','gate_abstention','no_context') and r['predicted_answer']=='Unanswerable'
            refusals[(label,'refuse' if refused else 'answer' if gen['status']=='ok' else 'failure')]+=1
    metrics={}
    for name in ('dense','bm25','hybrid','hybrid_rerank_pool','reranked'):
        metrics[name]={}
        for k in (5,10,20):
            rows=[ranking_metrics(r['retrieval']['rankings'][name],r['references'],k) for r in records]
            rows=[r for r in rows if r is not None]
            metrics[name][str(k)]={'eligible_questions':len(rows),**{m:mean([x[m] for x in rows]) for m in ('hit','union_evidence_recall','complete_reference_coverage','mrr')}}
    cm=[context_metrics(r['retrieval']['context'],r['references']) for r in records]
    return {'questions':len(records),'evidence_f1':mean(evidence),'retrieval':metrics,
            'context':{m:mean([x[m] for x in cm]) for m in ('gold_evidence_recall','complete_reference_coverage','contained_gold_character_fraction')},
            'avg_context_tokens':mean([r['retrieval']['context_tokens'] for r in records]),
            'generation':{'scheduled':len(gen_ids),'scored':len(answers),'statuses':dict(statuses),'answer_f1':mean(answers),
                          'answer_f1_by_best_reference_type':{k:{'n':len(v),'f1':mean(v)} for k,v in bytype.items()}},
            'gate_confusion':{f'{a}/{b}':n for (a,b),n in gates.items()},
            'answer_refusal_confusion':{f'{a}/{b}':n for (a,b),n in refusals.items()},
            'input_fidelity':'not_measured_preparsed_text'}
