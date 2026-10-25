"""Reproducible QASPER evaluation of ScientificRAG (pre-parsed text track).

No LLM calls unless --max_gen_queries is positive or -1. Never resume old v1
prediction files. Each run directory is bound to its immutable manifest.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bl_pipeline.rag.config import RAGConfig,load_token_counter
from bl_pipeline.rag.engine import ScientificRAG
from bl_pipeline.rag.llm_client import make_client,generate,GenerationResult,build_prompt
from bl_pipeline.rag.stage_wise_benchmark import references,summarize
from bl_pipeline.rag.gold_context import gold_context_trace,reader_diagnostic_eligible
from bl_pipeline.rag.schema import DocumentElement


def qasper_elements(paper_id, paper):
    elements = []
    def add(text, section, kind='paragraph', evidence=None):
        if not text or not text.strip(): return
        elements.append(DocumentElement(
            paper_id=paper_id, page=0, element_type=kind,
            text=text, section_path=section, reading_order=len(elements),
            metadata={'source_evidence': [evidence if evidence is not None else text],
                      'input_format': 'qasper_preparsed', 'page_provenance': 'unavailable'}
        ))
    add(paper.get('abstract', ''), 'Abstract')
    for i, sec in enumerate(paper.get('full_text', [])):
        section = f'{i}: {sec.get("section_name") or "Untitled"}'
        for para in sec.get('paragraphs', []): add(para, section)
    for i, ft in enumerate(paper.get('figures_and_tables', [])):
        caption = ft.get('caption', '')
        add(caption, f'Figure/table {i}', 'figure', f'FLOAT SELECTED: {caption}')
    return elements


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

def source_hash():
    base=Path(__file__).parent
    return digest({p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.rglob('*.py')) if 'tests' not in p.parts})

def bind_manifest(directory,manifest):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Run configuration/data/code changed. Use a NEW output directory; refusing mixed checkpoints.')
    else:
        if any(directory.iterdir()): raise ValueError('Output directory is nonempty without a v2 manifest')
        atomic_json(path,manifest)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev_path',default=str(ROOT/'data/benchmarks/qasper/qasper-dev-v0.3.json'))
    p.add_argument('--output_dir',required=True)
    p.add_argument('--tokenizer',required=True,help='hf:<reader tokenizer> or explicitly labeled tiktoken:<encoding> proxy')
    p.add_argument('--max_papers',type=int,default=0)
    p.add_argument('--max_gen_queries',type=int,default=0,help='0: no generation, -1: all, positive: seeded sample')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dense_model',default=RAGConfig.dense_model)
    p.add_argument('--reranker_model',default=RAGConfig.reranker_model)
    p.add_argument('--dense_revision'); p.add_argument('--reranker_revision')
    p.add_argument('--retrieval_k',type=int,default=30); p.add_argument('--rerank_k',type=int,default=30)
    p.add_argument('--evidence_k',type=int,default=5); p.add_argument('--context_budget',type=int,default=2048)
    p.add_argument('--gate_threshold',type=float,default=None)
    p.add_argument('--no_parent_expansion',action='store_true')
    p.add_argument('--oracle_gold_evidence',action='store_true',help='Budgeted gold-evidence reader diagnostic; excludes mixed/unanswerable and empty-evidence questions')
    p.add_argument('--llm_provider',choices=['groq','gemini','openrouter','openai'])
    p.add_argument('--llm_model')
    p.add_argument('--max_output_tokens',type=int,default=512)
    p.add_argument('--max_prompt_tokens',type=int,default=None,help='Optional prompt ceiling measured by the chosen tokenizer')
    p.add_argument('--request_delay',type=float,default=0)
    return p.parse_args()


def run(args, pipeline=None, client=None, token_counter=None):
    if args.max_papers<0 or args.max_gen_queries< -1: raise ValueError('Invalid sample limits')
    if args.max_output_tokens<=0 or args.request_delay<0: raise ValueError('Invalid generation configuration')
    data_path=Path(args.dev_path); data=json.loads(data_path.read_text(encoding='utf-8'))
    rng=random.Random(args.seed); pids=list(data)
    if args.max_papers: pids=sorted(rng.sample(pids,min(args.max_papers,len(pids))))
    queries=[(pid,qa) for pid in pids for qa in data[pid]['qas']]
    qids=[q['question_id'] for _,q in queries]
    if len(set(qids))!=len(qids): raise ValueError('Duplicate dataset question IDs')
    eligible_gen_ids=[qa['question_id'] for _,qa in queries
                      if not args.oracle_gold_evidence or reader_diagnostic_eligible(references(qa))]
    gen_ids=set(eligible_gen_ids if args.max_gen_queries==-1 else rng.sample(eligible_gen_ids,min(args.max_gen_queries,len(eligible_gen_ids))))
    if gen_ids and (not args.llm_model or not args.llm_provider): raise ValueError('Generation requires explicit --llm_provider and --llm_model')
    config=RAGConfig(dense_model=args.dense_model,reranker_model=args.reranker_model,
                     dense_revision=args.dense_revision,reranker_revision=args.reranker_revision,
                     retrieval_k=args.retrieval_k,rerank_k=args.rerank_k,evidence_k=args.evidence_k,
                     context_budget=args.context_budget,gate_threshold=args.gate_threshold,
                     expand_parents=not args.no_parent_expansion)
    versions={}
    for package in ('numpy','sentence-transformers','transformers','torch','tokenizers','tiktoken','openai'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    manifest={'schema_version':2,'pipeline':config.to_dict(),'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'source_sha256':source_hash(),'question_ids':qids,'generation_ids':sorted(gen_ids),'seed':args.seed,
              'tokenizer':args.tokenizer,'tokenizer_is_proxy':args.tokenizer.startswith('tiktoken:'),
              'oracle_gold_evidence':bool(args.oracle_gold_evidence),
              'gold_context_policy':'shortest_nonempty_reference_whole_units_v1' if args.oracle_gold_evidence else None,
              'generation_eligible_ids':eligible_gen_ids,
              'provider':args.llm_provider,'requested_model':args.llm_model,'max_output_tokens':args.max_output_tokens,
              'max_prompt_tokens':args.max_prompt_tokens,'packages':versions,'protocol':'qasper-all-annotations-floats-included',
              'retrieval_scope':'known_paper','prompt_template_sha256':digest(build_prompt('{question}','{context}'))}
    out=Path(args.output_dir)
    bind_manifest(out,manifest)
    lock=out/'RUNNING.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError('Run directory is in use; inspect RUNNING.lock before recovering an interrupted process')
    try:
        try: os.write(fd,str(os.getpid()).encode())
        finally: os.close(fd)
        counter=token_counter or load_token_counter(args.tokenizer)
        record_dir=out/'records'; record_dir.mkdir(exist_ok=True)
        run_id=digest(manifest); records=[]; count_children=0; count_parents=0; index_seconds=0.0
        # Initialize models only if an uncached query needs execution.
        for pid in pids:
            paper_q=data[pid]['qas']
            missing=any(not (record_dir/(digest(q['question_id'])+'.json')).exists() for q in paper_q)
            if missing and not args.oracle_gold_evidence:
                if pipeline is None: pipeline=ScientificRAG(config,counter)
                start=time.perf_counter()
                chunks=pipeline.index_elements(qasper_elements(pid,data[pid]))
                index_seconds+=time.perf_counter()-start
                count_parents+=sum(c.element_type=='parent' for c in chunks)
                count_children+=sum(c.element_type!='parent' for c in chunks)
            for qa in paper_q:
                qid=qa['question_id']; dest=record_dir/(digest(qid)+'.json')
                if dest.exists():
                    record=json.loads(dest.read_text(encoding='utf-8'))
                    checksum=record.pop('record_sha256',None)
                    if checksum!=digest(record) or record.get('run_id')!=run_id or record.get('question_id')!=qid:
                        raise ValueError('Checkpoint record failed integrity/configuration validation')
                    record['record_sha256']=checksum
                else:
                    if args.oracle_gold_evidence:
                        trace=gold_context_trace(references(qa),config.context_budget,counter)
                        retrieval_seconds=0.0
                    else:
                        start=time.perf_counter(); trace=pipeline.retrieve(qa['question'],paper_id=pid)
                        retrieval_seconds=time.perf_counter()-start
                    if trace['context_tokens']>config.context_budget: raise AssertionError('Context overflow')
                    prompt=build_prompt(qa['question'],trace['context']); prompt_tokens=counter(prompt)
                    gen=GenerationResult('not_requested',requested_model=args.llm_model or '')
                    gen_seconds=0.0
                    if qid in gen_ids:
                        if trace['gate']['abstain']:
                            gen=GenerationResult('gate_abstention',answer='Unanswerable',requested_model=args.llm_model)
                        elif not trace['context'] and not args.oracle_gold_evidence:
                            gen=GenerationResult('no_context',answer='Unanswerable',requested_model=args.llm_model)
                        elif args.max_prompt_tokens is not None and prompt_tokens>args.max_prompt_tokens:
                            gen=GenerationResult('prompt_overflow',requested_model=args.llm_model)
                        else:
                            if client is None: client=make_client(args.llm_provider)
                            start=time.perf_counter()
                            gen=generate(client,args.llm_model,prompt,max_tokens=args.max_output_tokens)
                            gen_seconds=time.perf_counter()-start
                            if args.request_delay: time.sleep(args.request_delay)
                    record={'schema_version':2,'run_id':run_id,'question_id':qid,'paper_id':pid,'question':qa['question'],
                            'references':references(qa),'retrieval':trace,'prompt':prompt,'prompt_sha256':digest(prompt),
                            'prompt_tokens':prompt_tokens,'generation':asdict(gen),'predicted_answer':gen.answer,
                            'predicted_evidence':trace['delivered_evidence'],'retrieval_seconds':retrieval_seconds,'generation_seconds':gen_seconds}
                    record['record_sha256']=digest(record); atomic_json(dest,record)
                records.append(record)
            print(f'Completed {len(records)}/{len(queries)} questions',flush=True)
        report=summarize(records,gen_ids)
        if args.oracle_gold_evidence:
            eligible_records=[r for r in records if r['question_id'] in eligible_gen_ids]
            complete_records=[r for r in eligible_records
                              if r['retrieval']['gold_diagnostic']['selected_reference_complete']]
            report['evaluation_condition']='budgeted_gold_evidence_reader_diagnostic'
            report['gold_diagnostic']={
                'eligible_questions':len(eligible_records),
                'excluded_questions':len(records)-len(eligible_records),
                'complete_reference_questions':len(complete_records),
                'incomplete_reference_questions':len(eligible_records)-len(complete_records),
                'complete_reference_generation':summarize(complete_records,gen_ids & {r['question_id'] for r in complete_records})['generation'],
                'note':'Gold-assisted diagnostic, not an end-to-end RAG score or a proven reader ceiling.'}
            # Gold labels generated this context: retrieval/evidence scores would be circular.
            report['retrieval']=None
            report['evidence_f1']=None
        report.update({'run_id':run_id,'papers':len(pids),'dataset_total_papers':len(data),
                       'complete_dataset':len(pids)==len(data),'tokenizer':args.tokenizer,
                       'context_overflow_count':sum(r['retrieval']['context_tokens']>config.context_budget for r in records),
                       'index_work_this_invocation':{'parents':count_parents,'children':count_children,'seconds':index_seconds}})
        atomic_json(out/'report.json',report)
        # All-question evidence export. Answer scores from this file include unrequested zeroes.
        temp=out/'predictions.jsonl.tmp'
        with temp.open('w',encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps({k:r[k] for k in ('question_id','predicted_answer','predicted_evidence')})+'\n')
        temp.replace(out/'predictions.jsonl')
        print(json.dumps(report,indent=2)); return report
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__': run(parse_args())
