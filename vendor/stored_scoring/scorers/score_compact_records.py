"""Offline stdlib-only scoring of a compact, full-precision stored-output ZIP."""
from array import array
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import zipfile

def require(value,message):
    if not value:raise ValueError(message)

def validate(p,tolerance):
    require(all(math.isfinite(x) and 0<=x<=1 for x in p),'invalid probability')
    require(abs(sum(p)-1)<=tolerance,'invalid probability sum')

def jev_guess(keys,p):
    # Native JevBench invalid outputs are wrong while keeping the denominator.
    # Its 0.02 rounding band is distinct from the MC bridge's 1e-5 rule.
    if not all(math.isfinite(x) and 0<=x<=1 for x in p) or abs(sum(p)-1)>.02:
        return None
    return min(zip(keys,p),key=lambda kv:(-kv[1],kv[0]))[0]

def mc_guess(probabilities,labels):
    require(set(labels)=={'contradiction','entailment','neutral'} and len(labels)==3,'invalid NLI label identity')
    for p in probabilities:validate(p,1e-5)
    entailment=[p[labels.index('entailment')] for p in probabilities]
    return max(range(len(entailment)),key=entailment.__getitem__)

def stats(values):
    mean=math.fsum(values)/len(values)
    sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
    return dict(n=len(values),mean=mean,descriptive_se=sd/math.sqrt(len(values)),per_setting=values)

def main(path):
    with zipfile.ZipFile(path) as z:
        population=json.loads(z.read('population.json'))
        manifest=json.loads(z.read('manifest.json'))
        experiments=json.loads(z.read('experiments.json'))
        provenance=json.loads(z.read('population_provenance.json'))
        occurrences=[]
        for block in population:
            for row in block['rows']:
                require(row['task'] in provenance,'missing source definition')
                occurrences.append((row['task'],row['source_ordinal'],row.get('qid')))
                require(row['candidate_occurrences']==list(range(len(row.get('keys',row.get('option_sha256'))))),'candidate occurrences not in original order')
        require(len(set(occurrences))==len(occurrences),'duplicate occurrence identity')
        results={}
        for model in manifest['models']:
            identifier=model['id'];values=array('d');values.frombytes(z.read('probabilities/'+identifier+'.f64le'))
            if sys.byteorder!='little':values.byteswap()
            cursor=0;counts=defaultdict(lambda:[0,0]);families=defaultdict(lambda:[0,0]);invalid_jev=0
            for block in population:
                for row in block['rows']:
                    task=row['task']
                    if block['kind']=='jk':
                        width=len(row['keys']);p=values[cursor:cursor+width];cursor+=width
                        require(len(p)==width,'truncated JK values')
                        if task=='jevbench_public':
                            guess=jev_guess(row['keys'],p);invalid_jev+=guess is None
                            correct=int(guess==row['gold']);families[row['family']][0]+=correct;families[row['family']][1]+=1
                        else:
                            validate(p,max(1e-5,width*.005+1e-8))
                            if row['variant']!='clean' or row['source']=='unknowable':continue
                            guess=row['keys'][max(range(width),key=p.__getitem__)]
                            correct=int(guess==row['gold'])
                    else:
                        require(row['scoring_call_role']=='rerank','gold-conditioned calls cannot be scored as reranking')
                        width=len(row['option_sha256']);p=values[cursor:cursor+3*width];cursor+=3*width
                        require(len(p)==3*width,'truncated MC values')
                        correct=int(mc_guess([p[start:start+3] for start in range(0,len(p),3)],block['probability_labels'])==row['gold'])
                    counts[task][0]+=correct;counts[task][1]+=1
            require(cursor==len(values),'unconsumed probability values')
            require(len(families)==18 and sum(v[1] for v in families.values())==231,'JevBench population differs')
            tasks={task:100*c/n for task,(c,n) in counts.items()}
            tasks['jevbench_public']=100*math.fsum(c/n for c,n in families.values())/18
            require(len(tasks)==18,'task population differs')
            reference=json.loads(z.read('reference/'+identifier+'.json'))
            for task,target in reference.items():
                require(abs(tasks[task]-target['score_percent'])<1e-10,'native reference mismatch: '+identifier+'/'+task)
                if task=='jevbench_public':
                    require({k:dict(n_correct=c,n_scorable=n) for k,(c,n) in families.items()}==target['families'],'native family integer mismatch')
                else:
                    n=target['clean_questions'] if task.startswith('kev_') else target['questions']
                    require(counts[task][1]==n and counts[task][0]==round(target['score_percent']*n/100),'native task integer mismatch')
            endpoints={'JevBench':tasks['jevbench_public'],
                       'Kev':math.fsum(v for t,v in tasks.items() if t.startswith('kev_'))/6,
                       'MC-11':math.fsum(v for t,v in tasks.items() if t!='jevbench_public' and not t.startswith('kev_'))/11}
            results[identifier]=dict(tasks_percent=tasks,endpoints_percent=endpoints,correct_denominator=dict(counts),families=dict(families),invalid_jev_questions=invalid_jev)
        derived={}
        for name,experiment in experiments.items():
            entry={'absolute':{},'contrasts':{}}
            settings=experiment['settings']
            arms=set(settings[0]['models'])
            for arm in sorted(arms):
                entry['absolute'][arm]={f:{m:stats([results[s['models'][arm]][f][m] for s in settings])
                    for m in next(iter(results.values()))[f]} for f in ['tasks_percent','endpoints_percent']}
            for label,coeff in experiment['contrast_coefficients'].items():
                entry['contrasts'][label]={f.replace('percent','pp'):{m:stats([
                    math.fsum(c*results[s['models'][arm]][f][m] for arm,c in coeff.items()) for s in settings])
                    for m in next(iter(results.values()))[f]} for f in ['tasks_percent','endpoints_percent']}
            derived[name]=entry
        return dict(status='PASS',models=len(results),tasks=len(results)*18,results=results,derived=derived,
                    scope=manifest['scope'],precision='Original JSON float values stored losslessly as binary64, no rounding/quantization.')

if __name__=='__main__':
    report=main(sys.argv[1])
    output=Path(sys.argv[2]) if len(sys.argv)>2 else Path('compact_reconstructed_scores.json')
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],models=report['models'],tasks=report['tasks'],output=str(output))))
