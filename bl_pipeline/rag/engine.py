"""Single supported RAG engine: typed chunks, independent sparse/dense retrieval,
reranking, explicit gate, and budgeted focal evidence plus section expansion.
No agent loop, Chroma, Nemotron service, or vision-LLM cascade is used.
"""
from dataclasses import asdict
from collections import Counter
import math
import re
import uuid
import numpy as np
from bl_pipeline.rag.config import RAGConfig
from bl_pipeline.rag.structured_chunker import StructurePreservingChunker
from bl_pipeline.rag.token_budgeter import SemanticTokenBudgeter, RetrievedFocalItem
from bl_pipeline.rag.crag_gate import RetrievalQualityGate

class DocumentBM25:
    def __init__(self, corpus):
        self.tf=[Counter(re.findall(r'\w+|[^\w\s]',s.lower())) for s in corpus]
        self.lengths=[sum(x.values()) for x in self.tf]
        self.avg=sum(self.lengths)/max(len(corpus),1) or 1
        self.df=Counter(t for terms in self.tf for t in terms)
    def score(self, query):
        tokens=[t for t in re.findall(r'\w+|[^\w\s]',(query or '').lower()) if t not in {'?','!',',','.',';',':','"',"'"}]; out=[]
        if not tokens or not self.tf: return [0.0]*len(self.tf)
        for terms,length in zip(self.tf,self.lengths):
            out.append(sum(math.log(1+(len(self.tf)-self.df[t]+.5)/(self.df[t]+.5))*terms[t]*2.5/
                           (terms[t]+1.5*(.25+.75*length/self.avg))
                           for t in tokens if terms[t]))
        return out

class ScientificRAG:
    def __init__(self, config, token_counter, dense_model=None, reranker=None):
        self.config=config
        self.budgeter=SemanticTokenBudgeter(config.context_budget,token_counter)
        self.gate=RetrievalQualityGate(config.gate_threshold)
        if dense_model is None:
            from sentence_transformers import SentenceTransformer
            dense_model=SentenceTransformer(config.dense_model,revision=config.dense_revision)
        if reranker is None:
            from sentence_transformers import CrossEncoder
            reranker=CrossEncoder(config.reranker_model,revision=config.reranker_revision)
        self.dense=dense_model; self.reranker=reranker
        self.chunks=[]; self.parents={}; self.vectors=None

    def index_elements(self, elements):
        # One chunker used for both PDF and pre-parsed dataset inputs.
        papers={}
        for element in elements: papers.setdefault(element.paper_id,[]).append(element)
        chunks=[c for paper_id,group in papers.items()
                for c in StructurePreservingChunker().chunk_elements(group,paper_id=paper_id)]
        mapping={}
        for paper_id in papers:
            paper_chunks=[c for c in chunks if c.paper_id==paper_id]
            for i,c in enumerate(paper_chunks):
                mapping[c.chunk_id]=str(uuid.uuid5(uuid.NAMESPACE_URL,f'{c.paper_id}:{c.section_path}:{i}:{c.content}'))
        for c in chunks:
            old=c.chunk_id; c.chunk_id=mapping[old]
            c.parent_id=mapping.get(c.parent_id,c.parent_id)
        parents={c.chunk_id:c for c in chunks if c.element_type=='parent'}
        children=[c for c in chunks if c.element_type!='parent']
        for c in children:
            if c.parent_id not in parents: raise ValueError('Orphan child')
        self.parents=parents; self.chunks=children
        self.vectors=(np.asarray(self.dense.encode([self._embedding_text(c) for c in children],
                      normalize_embeddings=True,show_progress_bar=False)) if children else np.empty((0,0)))
        return chunks

    @staticmethod
    def _embedding_text(c): return f'Section: {c.section_path}\n{c.content}'

    def retrieve(self, query, paper_id=None):
        cfg=self.config
        eligible=[i for i,c in enumerate(self.chunks) if paper_id is None or c.paper_id==paper_id]
        if not eligible:
            return {'rankings':{k:[] for k in ('dense','bm25','hybrid','hybrid_rerank_pool','reranked')},'selected_evidence':[],
                    'delivered_evidence':[],'selected_ids':[],'context':'','context_tokens':0,
                    'omitted_ids':[],'gate':asdict(self.gate.evaluate_retrieval(query,[]))}
        # Filter before both modalities: no unscoped lexical escape.
        texts=[self._embedding_text(self.chunks[i]) for i in eligible]
        bm25=DocumentBM25(texts).score(query)
        q=np.asarray(self.dense.encode([cfg.query_instruction+query],normalize_embeddings=True,show_progress_bar=False))[0]
        sims=self.vectors[eligible]@q
        rank=lambda scores: [eligible[j] for j in sorted(range(len(eligible)),key=lambda j:(-float(scores[j]),j))[:cfg.retrieval_k]]
        dense=rank(sims); sparse=rank(bm25)
        # Do not award sparse RRF points to zero-overlap candidates.
        sparse=[i for i in sparse if bm25[eligible.index(i)]>0]
        rrf=Counter()
        for ids in (dense,sparse):
            for r,i in enumerate(ids,1): rrf[i]+=1/(60+r)
        hybrid=sorted(rrf,key=lambda i:(-rrf[i],i))[:cfg.retrieval_k]
        pool=hybrid[:cfg.rerank_k]
        scores=self.reranker.predict([(query,self._embedding_text(self.chunks[i])) for i in pool])
        if len(scores)!=len(pool) or not np.isfinite(scores).all(): raise ValueError('Invalid reranker output')
        ordered=sorted(zip(pool,scores),key=lambda x:(-float(x[1]),x[0]))
        reranked=[i for i,_ in ordered]
        gate=self.gate.evaluate_retrieval(query,[{'rerank_score':float(s)} for _,s in ordered])
        selected=[] if gate.abstain else ordered[:cfg.evidence_k]
        items=[]
        for i,score in selected:
            c=self.chunks[i]; parent=self.parents[c.parent_id]
            items.append(RetrievedFocalItem(c.chunk_id,c.paper_id,c.element_type,c.content,float(score),
                         c.section_path,c.page,parent_section_text=parent.content if cfg.expand_parents else ''))
        packed=self.budgeter.pack(items,query)
        keys=lambda indices:list(dict.fromkeys(key for i in indices for key in self.chunks[i].metadata.get('source_evidence',[self.chunks[i].content])))
        # Background expansion is not automatically a predicted supporting citation.
        allkeys=keys([i for i,_ in selected])
        visible=lambda e:e.removeprefix('FLOAT SELECTED: ').strip() in packed.text
        delivered=[e for e in allkeys if e.strip() and visible(e)]
        return {'rankings':{name:[{'chunk_id':self.chunks[i].chunk_id,'evidence':keys([i])} for i in ids]
                           for name,ids in [('dense',dense),('bm25',sparse),('hybrid',hybrid),('hybrid_rerank_pool',pool),('reranked',reranked)]},
                'selected_evidence':keys([i for i,_ in selected]),'delivered_evidence':delivered,
                'selected_ids':[self.chunks[i].chunk_id for i,_ in selected],
                'rerank_scores':[float(s) for _,s in ordered],
                'context':packed.text,'context_tokens':packed.token_count,'omitted_ids':packed.omitted_ids,
                'gate':asdict(gate)}

# The former RAGEngine was removed. No legacy compatibility implementation is retained.
