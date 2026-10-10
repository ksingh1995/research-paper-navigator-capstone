from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from . import hybrid, agent, safeguards

ROOT = Path(__file__).resolve().parents[2]

def default_corpus():
    configured = os.getenv('RPN_CORPUS_DIR')
    if configured:
        return Path(configured)
    full = ROOT/'data/papers'
    return full if any(full.glob('*.pdf')) else ROOT/'data/sample_papers'

def corpus_fingerprint(chunks):
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode('utf8'))
        digest.update(b'\0')
        digest.update(chunk.text.encode('utf8'))
        digest.update(b'\0')
    return digest.hexdigest()

def named_papers(question, chunks):
    """Derive title anchors from the question, rather than test-set answer labels."""
    q = agent.title_key(question)
    titles = sorted({c.title for c in chunks}, key=len, reverse=True)
    found, keys = [], []
    for title in titles:
        words = title.split()
        anchor = title
        key = agent.title_key(anchor)
        if key not in q:
            # PDF extraction can append an author to the title, e.g. "... REPL Kevin Ellis*".
            # Match a long leading title phrase, dropping at most three trailing words.
            for drop in range(1,min(3,len(words)-4)+1):
                prefix = ' '.join(words[:-drop])
                prefix_key = agent.title_key(prefix)
                if len(prefix_key.split()) >= 4 and len(prefix_key.split())/len(key.split()) >= .65 and prefix_key in q:
                    anchor,key = prefix,prefix_key
                    break
        if len(key.split()) < 4 or key not in q or any(key in prior or prior in key for prior in keys):
            continue
        found.append(anchor)
        keys.append(key)
    return found[:2]

class Navigator:
    """One loaded corpus and sequential queries. Module configuration is process-scoped."""
    def __init__(self, corpus: Path, cache_dir: Path, *, model=None):
        self.corpus = corpus.resolve()
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        hybrid.CACHE_PATH = self.cache_dir / 'chunks.json.gz'
        hybrid.EMBEDDING_CACHE_PATH = self.cache_dir / 'embeddings.npz'
        safeguards.MODEL = model or os.getenv('RPN_CHAT_MODEL', 'openai/gpt-5.4-mini')
        self.chunks, self.warnings, self.pdf_count = hybrid.prepare_chunks(self.corpus)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.fingerprint = corpus_fingerprint(self.chunks)
        self.retriever = None
        self.cache_file = self.cache_dir / 'answers.json'
        try:
            self.cache = json.loads(self.cache_file.read_text(encoding='utf8'))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.cache = {}
        if not isinstance(self.cache, dict):
            self.cache = {}

    def keyword(self, question, k=5):
        clean, notes = safeguards.sanitize_question(question)
        hits = hybrid.BM25Retriever(self.chunks).search(clean, k=k)
        return {'mode': 'keyword_only', 'question': clean, 'filter_notes': notes,
                'pdf_count': self.pdf_count, 'chunks': len(self.chunks),
                'extraction_warnings': self.warnings, 'hits': [asdict(h) for h in hits]}

    def ask(self, question, *, use_cache=True):
        start = time.perf_counter()
        clean, notes = safeguards.sanitize_question(question)
        if len(clean) < 12:
            return {'mode': 'live', 'route': 'rejected', 'answer': 'Please provide a clear research-paper question.',
                    'filter_notes': notes, 'calls': [], 'hits': []}
        cached = safeguards.cache_lookup(self.cache, clean, self.fingerprint) if use_cache else None
        if isinstance(cached, str) and cached:
            obj = self.cache[safeguards.cache_key(clean, self.fingerprint)]
            hits = obj.get('hits', [])
            return {'mode': 'live', 'route': 'exact cache hit', 'answer': cached,
                    'sanitized_question': clean, 'filter_notes': notes, 'hits': hits, 'calls': [],
                    'metrics': safeguards.metrics([], time.perf_counter()-start, {'hits': hits}),
                    'corpus_fingerprint': self.fingerprint}
        hybrid.load_openrouter_key()
        meter = safeguards.Meter(hybrid)
        original_post = hybrid.openrouter_post
        hybrid.openrouter_post = meter
        try:
            if self.retriever is None:
                vectors, index_usage = hybrid.load_or_build_semantic_index(self.chunks)
                self.retriever = hybrid.HybridRetriever(self.chunks, hybrid.BM25Retriever(self.chunks), vectors)
            else:
                index_usage = {'source': 'loaded_in_memory'}
            index_calls = list(meter.calls)
            meter.calls = []
            production_start = time.perf_counter()
            route = safeguards.route_question(clean)
            titles = named_papers(clean, self.chunks)
            if len(titles) > 1:
                route = 'agent'
            if route == 'direct':
                result = safeguards.run_direct(hybrid, self.retriever, self.by_id, self.by_id, clean)
            else:
                result = safeguards.run_hardened_agent(agent, hybrid, self.retriever, self.chunks,
                                                      self.by_id, self.by_id, clean, titles)
            result.update(mode='live', route=route, question=question, filter_notes=notes,
                          named_papers=titles, calls=meter.calls,
                          metrics=safeguards.metrics(meter.calls,time.perf_counter()-production_start,result),
                          index_usage=index_usage, index_calls=index_calls,
                          corpus_fingerprint=self.fingerprint, extraction_warnings=self.warnings,
                          total_seconds=round(time.perf_counter()-start,3))
            if use_cache and route == 'direct' and result.get('hits') and not result.get('rejected_chunks'):
                if not re.search(r'\b(he|she|they|this|that|it|those|former|latter)\b',clean,re.I):
                    self.cache[safeguards.cache_key(clean,self.fingerprint)] = {
                        'created': time.time(), 'question': clean, 'answer': result['answer'],
                        'hits': result['hits']}
                    self.cache_file.write_text(json.dumps(self.cache,indent=2,ensure_ascii=False),encoding='utf8')
            return result
        finally:
            hybrid.openrouter_post = original_post

def evidence_summary():
    weeks={i:json.loads((ROOT/f'evidence/week{i}_results.json').read_text(encoding='utf8')) for i in range(3,8)}
    totals = {}
    for arm in ('baseline','hardened'):
        records = [r for r in weeks[7]['records'] if r['arm']==arm]
        totals[arm] = {'cases':len(records), **{k:round(sum(r['metrics'][k] for r in records),8)
                  for k in ('total_tokens','seconds','reported_cost_usd')}}
    savings = {k:round(100*(1-totals['hardened'][k]/totals['baseline'][k]),1)
               for k in ('total_tokens','seconds','reported_cost_usd')}
    return {**{f'week{i}':weeks[i]['summary'] for i in range(3,7)},'week7':totals,'week7_reduction_percent':savings,
            'note':'Different weeks used different questions. Within-week paired comparisons only. Week 7 has seven single-run pairs plus an excluded cache repeat. Historical results, not new live results.'}

def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf8',errors='replace')
    parser = argparse.ArgumentParser(description='Research Paper Navigator final capstone')
    sub = parser.add_subparsers(dest='command',required=True)
    live = sub.add_parser('ask',help='Live hybrid retrieval and a grounded answer (uses OpenRouter)')
    live.add_argument('question')
    live.add_argument('--corpus',type=Path,default=default_corpus())
    live.add_argument('--cache-dir',type=Path,default=ROOT/'.cache')
    live.add_argument('--no-cache',action='store_true')
    live.add_argument('--model',default=None)
    live.add_argument('--save',type=Path,help='Save complete answer, traces and measurements as JSON')
    keyword = sub.add_parser('keyword',help='Local BM25 evidence search, no key or network')
    keyword.add_argument('question')
    keyword.add_argument('--corpus',type=Path,default=default_corpus())
    keyword.add_argument('--cache-dir',type=Path,default=ROOT/'.cache')
    keyword.add_argument('--save',type=Path)
    replay = sub.add_parser('replay',help='Display a recorded Week 7 answer, without API calls')
    replay.add_argument('--case',default='P1',choices=['P1','P2','Q1','S1','S3','S4_poison','S5'])
    replay.add_argument('--arm',default='hardened',choices=['baseline','hardened'])
    sub.add_parser('summary',help='Recalculate the historical evidence summaries')
    sub.add_parser('check',help='Run deterministic safeguard checks, without API calls')
    args = parser.parse_args()
    try:
        if args.command == 'check':
            safeguards.offline_checks()
            return
        if args.command == 'summary':
            result = evidence_summary()
        elif args.command == 'replay':
            data = json.loads((ROOT/'evidence/week7_results.json').read_text(encoding='utf8'))
            result = next(r for r in data['records'] if r['case_id']==args.case and r['arm']==args.arm)
            print('RECORDED DEMONSTRATION: saved Week 7 output, not a new live generation.\n')
            print('Question:',result['question'],'\n\n'+result['answer'])
            print('\nRecorded route:',result['route'])
            print('Recorded metrics:',json.dumps(result['metrics'],ensure_ascii=False))
            return
        else:
            nav = Navigator(args.corpus,args.cache_dir,model=getattr(args,'model',None))
            result = nav.ask(args.question,use_cache=not args.no_cache) if args.command=='ask' else nav.keyword(args.question)
            if args.command=='ask':
                print(result['answer'])
                print('\nRoute:',result['route'],'\nMetrics:',json.dumps(result.get('metrics',{})))
            else:
                print(json.dumps(result,indent=2,ensure_ascii=False))
            if args.save:
                args.save.parent.mkdir(parents=True,exist_ok=True)
                args.save.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
            return
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (FileNotFoundError,RuntimeError,ValueError,OSError) as exc:
        parser.exit(1,f'Error: {exc}\n')
