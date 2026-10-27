"""Targeted native-semantics edge cases and actual frozen duplicate records."""
import json
from pathlib import Path
import sys
import zipfile
from score_compact_records import jev_guess,mc_guess,validate

def check(path):
    assert jev_guess(['z','a'],[.5,.5])=='a'  # Native lexical order, not option order.
    assert jev_guess(['a','z'],[.5-1e-12,.5+1e-12])=='z'  # No epsilon ties.
    assert jev_guess(['a','z'],[.2,.2]) is None  # Invalid is wrong, no denominator deletion.
    assert jev_guess(['a','z'],[float('nan'),.5]) is None
    labels=['neutral','entailment','contradiction']
    assert mc_guess([[.4,.5,.1],[.4,.5,.1]],labels)==0  # Original candidate first max.
    assert mc_guess([[.1,.7,.2],[.2,.6,.2]],['entailment','neutral','contradiction'])==1
    assert mc_guess([[.25,.5-1e-12,.25+1e-12],[.25,.5+1e-12,.25-1e-12]],labels)==1
    try:mc_guess([[.3,.3,.3]],labels)
    except ValueError:pass
    else:raise AssertionError('Invalid MC sum must fail, not be renormalized/deleted')
    with zipfile.ZipFile(path) as z:population=json.loads(z.read('population.json'))
    mmlu=next(b['rows'] for b in population if b['name']=='mc_questions/mmlu.jsonl')
    assert mmlu[10324]['question_sha256']==mmlu[10325]['question_sha256']
    assert mmlu[10324]['occurrence_id']!=mmlu[10325]['occurrence_id']
    assert mmlu[10324]['base_question_id']!=mmlu[10325]['base_question_id']
    for ordinal in [2178,4021,6494,13601]:
        row=mmlu[ordinal];gold=row['gold'];assert row['option_sha256'].count(row['option_sha256'][gold])>1
        assert 'source answer index' in row['gold_source']
    jk=next(b['rows'] for b in population if b['kind']=='jk')
    records={};count=0
    for row in jk:
        if row['task']=='kev_decision-v7_development':records.setdefault(row['record_ordinal'],[]).append(row);count+=1
    assert len(records)==1204 and count>1204
    assert any(len({x['qid'] for x in rows})>1 for rows in records.values())
    return dict(status='PASS',cases=['native lexical Jev tie','exact non-tie','Jev invalid counts wrong','MC original-order exact tie','NLI label identity','MC invalid fails','adjacent duplicate MMLU occurrences','four duplicate-gold slots','Kev record versus question identity'])

if __name__=='__main__':print(json.dumps(check(sys.argv[1])))
