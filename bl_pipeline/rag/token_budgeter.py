"""Whole-unit context packing with an enforced serialized token budget."""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrievedFocalItem:
    item_id: str
    paper_id: str
    element_type: str
    content: str
    score: float = 0.0
    section_path: str = ''
    page: int = 1
    equation_ref: str | None = None
    table_ref: str | None = None
    parent_section_text: str = ''
    cross_reference_texts: list[str] = field(default_factory=list)
    sibling_texts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PackedContext:
    text: str
    token_count: int
    included_ids: list[str]
    omitted_ids: list[str]

class SemanticTokenBudgeter:
    def __init__(self, total_token_budget=2048, token_counter=None):
        if total_token_budget <= 0: raise ValueError('Budget must be positive')
        if token_counter is None: raise ValueError('An explicit reader tokenizer/counter is required')
        self.total_budget=total_token_budget
        self.count_tokens=token_counter
        self.last_result=None

    @staticmethod
    def _render(items, extras):
        blocks=[]
        for item in items:
            header=f'[Source {item.item_id}; paper={item.paper_id}; page={item.page or "unknown"}; section={item.section_path}]'
            content = (item.content or '').strip()
            blocks.append(header+'\n'+content+''.join('\n\n'+x for x in extras.get(item.item_id, [])))
        return '\n\n---\n\n'.join(blocks)

    def pack(self, focal_items, query=''):
        accepted=[]; omitted=[]; extras={}
        items = focal_items or []
        # Admit primary evidence before spending any budget on parent expansion.
        for item in items:
            content = (item.content or '').strip()
            if not content or any(x.item_id==item.item_id for x in accepted): continue
            if self.count_tokens(self._render(accepted+[item],extras)) <= self.total_budget:
                accepted.append(item)
            else: omitted.append(item.item_id)
        used_texts={(x.content or '').strip() for x in accepted}
        for item in accepted:
            expansions = (item.cross_reference_texts or []) + (item.sibling_texts or []) + (item.parent_section_text or '').split('\n\n')
            for text in expansions:
                if text is None: continue
                if not isinstance(text, str):
                    raise TypeError('Context expansions must be strings or None')
                text=text.strip()
                if not text or text in used_texts or any(text in (x.content or '') for x in accepted): continue
                trial={k:list(v) for k,v in extras.items()}
                trial.setdefault(item.item_id,[]).append(text)
                if self.count_tokens(self._render(accepted,trial)) <= self.total_budget:
                    extras=trial; used_texts.add(text)
        text=self._render(accepted,extras)
        result=PackedContext(text,self.count_tokens(text),[x.item_id for x in accepted],omitted)
        if result.token_count > self.total_budget: raise AssertionError('Context budget exceeded')
        self.last_result=result
        return result

    def format_calibrated_context(self, focal_items, query=''):
        return self.pack(focal_items,query).text
