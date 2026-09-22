"""Export full review items and import explicit, reasoned LLM judgments."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, digest, read_jsonl, require, write_json, write_jsonl

RUBRIC = """Adjudicate only instruction, context, reference, easy response and candidate hard responses.
Never inspect GPT-2 scores or generated outputs when selecting negatives.
Gold must answer the prompt with no material factual, contextual, or instruction-constraint error.
Easy negative must fail to answer the prompt. A hard negative must also be incorrect and share
a meaningful subject or closely related task/answer space, not merely a word or broad dataset label.
For subjective tasks accept reasonable alternatives; reject negatives that would also answer the prompt.
Select the best qualifying frozen hard option. If none qualifies, reject rather than invent a replacement.
Record separate booleans, an item-specific reason, and uncertainty. False or uncertain items are excluded
from the vetted ranking set. Do not modify references. All judgments are LLM judgments, not human labels.
"""


def show(root, start, end):
    test = {r['id']: r for r in read_jsonl(root / 'data/test.jsonl')}
    candidates = read_jsonl(root / 'data/candidates.jsonl')
    for i in range(start, min(end, len(candidates))):
        c = candidates[i]
        r = test[c['id']]
        print(f"\nITEM {i} {r['id']} {r['category']}")
        for label, value in [('INSTRUCTION', r['instruction']), ('CONTEXT', r['context']),
                             ('GOLD', r['response']), ('EASY', test[c['easy_id']]['response'])]:
            print(label, value)
        for option in c['hard_options']:
            other = test[option['id']]
            print('HARD', other['id'], other['response'])


def apply(root, decisions_path):
    candidates = read_jsonl(root / 'data/candidates.jsonl')
    reviews = read_jsonl(root / 'review/negatives.jsonl')
    decisions = read_jsonl(decisions_path)
    by_id = {r['id']: r for r in decisions}
    require(len(by_id) == len(decisions) == len(candidates), 'Need exactly one judgment per candidate')
    require(set(by_id) == {r['id'] for r in candidates}, 'Judgment IDs differ from frozen pool')
    candidate_map = {c['id']: c for c in candidates}
    for r in reviews:
        d = by_id[r['id']]
        require(all(type(d.get(k)) is bool for k in ['gold_valid', 'easy_valid', 'hard_valid']), 'Missing boolean decision')
        require(d.get('confidence') in {'high', 'medium', 'low'} and d.get('notes', '').strip(), 'Missing rationale or confidence')
        selected = d.get('hard_id', r['hard_id'])
        require(selected in {x['id'] for x in candidate_map[r['id']]['hard_options']}, 'Hard candidate not in frozen pool')
        require(not r.get('reviewer', '').strip() and not r.get('adjudicator', '').strip(), 'Existing judgments must be preserved, not overwritten')
        r.update(d)
        r.update(hard_id=selected, adjudicator='Codex (GPT-6), current task', adjudicator_type='llm',
                 disposition='include' if all(d[k] for k in ['gold_valid', 'easy_valid', 'hard_valid']) and d['confidence'] != 'low' else 'exclude')
    write_jsonl(root / 'review/negatives.before_llm.jsonl', read_jsonl(root / 'review/negatives.jsonl'))
    write_jsonl(root / 'review/negatives.jsonl', reviews)
    write_json(root / 'review/adjudication_manifest.json', {
        'adjudicator_type': 'llm', 'adjudicator': 'Codex (GPT-6), current task',
        'rubric': RUBRIC, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'candidate_hash': digest(candidates), 'decisions_hash': digest(decisions), 'review_hash': digest(reviews),
        'reviewed': len(reviews), 'included': sum(r['disposition'] == 'include' for r in reviews),
        'selection_before_model_evaluation': True, 'external_judge_api_used': False,
        'limits': 'One LLM adjudicator, no independent human validation; factual uncertainty can remain.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'artifacts')
    sub = parser.add_subparsers(dest='command', required=True)
    display = sub.add_parser('show')
    display.add_argument('start', type=int)
    display.add_argument('end', type=int)
    importer = sub.add_parser('apply')
    importer.add_argument('decisions', type=Path)
    args = parser.parse_args()
    if args.command == 'show':
        show(args.root, args.start, args.end)
    else:
        apply(args.root, args.decisions)
