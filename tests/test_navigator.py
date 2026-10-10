"""Offline tests exercise the production wrapper without a provider or paper files."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_navigator import hybrid,agent,safeguards
from research_navigator.cli import Navigator,named_papers,evidence_summary

def chunk(i,title,text):
    return hybrid.Chunk(f'paper{i}:p1:c1',f'paper{i}.pdf',title,1,1,text)

class SystemTests(unittest.TestCase):
    def test_safeguards(self):
        safeguards.offline_checks()

    def test_changed_passage_is_rejected(self):
        from dataclasses import replace
        original=chunk(1,'Known Paper','Authentic text')
        rejected=[]
        self.assertIsNone(safeguards.trusted_chunk(replace(original,text='fabricated'),{original.chunk_id:original},rejected))
        self.assertEqual(rejected,[original.chunk_id])

    def test_hybrid_retains_component_ranks(self):
        chunks=[chunk(1,'Learning to Infer Program Sketches','neural sketch symbolic synthesis'),chunk(2,'Other Paper','protein structure')]
        vectors=np.array([[1.0,0.0],[0.0,1.0]],dtype=np.float32)
        with patch.object(hybrid,'embed_texts',return_value=(np.array([[1.0,0.0]],dtype=np.float32),{'tokens':2,'cost':0})):
            hits,_=hybrid.HybridRetriever(chunks,hybrid.BM25Retriever(chunks),vectors).search('neural sketch',k=2)
        self.assertEqual(hits[0].chunk_id,chunks[0].chunk_id)
        self.assertIsNotNone(hits[0].keyword_rank)
        self.assertIsNotNone(hits[0].semantic_rank)

    def test_direct_wrapper_and_cache(self):
        chunks=[chunk(1,'Learning to Infer Program Sketches','Neural sketches leave holes for symbolic search.')]
        vectors=np.zeros((1,hybrid.EMBEDDING_DIMENSIONS),dtype=np.float32)
        vectors[0,0]=1
        calls=[]
        def fake_post(path,body,**kwargs):
            calls.append(path)
            if path=='/embeddings':
                return {'data':[{'index':0,'embedding':vectors[0].tolist()}],'usage':{'total_tokens':4,'cost':0}}
            return {'choices':[{'message':{'content':'The neural model proposes sketches and symbolic search completes holes [1].'}}],
                    'usage':{'prompt_tokens':20,'completion_tokens':12,'total_tokens':32,'cost':0}}
        with tempfile.TemporaryDirectory() as d, patch.object(hybrid,'prepare_chunks',return_value=(chunks,[],1)), \
             patch.object(hybrid,'load_openrouter_key',return_value='test-only'), \
             patch.object(hybrid,'load_or_build_semantic_index',return_value=(vectors,{'source':'mock'})), \
             patch.object(hybrid,'openrouter_post',side_effect=fake_post):
            nav=Navigator(Path(d),Path(d)/'cache')
            first=nav.ask(safeguards.METHOD_Q)
            second=nav.ask(safeguards.METHOD_Q)
            self.assertEqual(first['route'],'direct')
            self.assertEqual(first['metrics']['chat_calls'],1)
            self.assertEqual(second['route'],'exact cache hit')
            self.assertEqual(first['answer'],second['answer'])
            self.assertEqual(first['hits'],second['hits'])
            self.assertEqual(second['metrics']['api_calls'],0)
            self.assertEqual(calls,['/embeddings','/chat/completions'])

    def test_agent_routes_two_named_papers_without_gold_labels(self):
        chunks=[chunk(1,'Learning to Infer Program Sketches','Neural symbolic sketches experimental evaluation scope.'),
                chunk(2,'Program Synthesis with a REPL','Execution feedback repl evaluation limitations.')]
        titles=named_papers(safeguards.MULTI_Q,chunks)
        self.assertEqual(set(titles),{c.title for c in chunks})
        hit=lambda c: hybrid.HybridSearchResult(1,0.1,1,2.0,1,0.9,c.chunk_id,c.filename,c.title,c.page,c.text)
        class MockRetriever:
            def search(self,q,k=5): return [hit(c) for c in chunks],{'tokens':2,'cost':0}
        calls=[]
        def fake_chat(b,s,u,max_tokens):
            calls.append(s)
            return ('{"action":"answer"}' if s==safeguards.PLANNER_SYSTEM else 'Comparison with citations [1][2].'),{}
        by_id={c.chunk_id:c for c in chunks}
        with patch.object(safeguards,'call_chat',side_effect=fake_chat):
            result=safeguards.run_hardened_agent(agent,hybrid,MockRetriever(),chunks,by_id,by_id,safeguards.MULTI_Q,titles)
        self.assertEqual(len(result['actions']),3)
        self.assertEqual({a['title'] for a in result['actions'] if a['action']=='lookup_paper'},set(titles))
        self.assertEqual(len(calls),3)

    def test_expiry_and_corpus_changes_miss(self):
        import time
        q=safeguards.METHOD_Q
        cache={safeguards.cache_key(q,'corpus1'):{'created':time.time()-safeguards.CACHE_TTL_SECONDS-1,'answer':'old'}}
        self.assertIsNone(safeguards.cache_lookup(cache,q,'corpus1'))
        self.assertIsNone(safeguards.cache_lookup(cache,q,'corpus2'))

    def test_title_with_extracted_author_suffix(self):
        chunks=[chunk(1,'Program Synthesis with a REPL Kevin Ellis*','Method text.')]
        self.assertEqual(named_papers(safeguards.MULTI_Q,chunks),['Program Synthesis with a REPL'])

    def test_historical_summary(self):
        s=evidence_summary()
        self.assertEqual(s['week7']['baseline']['cases'],7)
        self.assertEqual(s['week7']['hardened']['total_tokens'],40115)
        self.assertEqual(s['week7_reduction_percent']['total_tokens'],41.6)

if __name__=='__main__': unittest.main()
