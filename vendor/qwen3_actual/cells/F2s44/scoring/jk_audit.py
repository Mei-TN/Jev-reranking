"""Full native seven-population and qtype audit of already produced probabilities."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from common import durable, read, sha, worker_alarm
from population import canonical, counter_digest, request_digest, request_token_digest

JK_COUNTS = {'jevbench_public':231, 'kev_decision-v7_development':1204,
 'kev_decision-v7_test':1176, 'kev_transfer-v4_development':764,
 'kev_transfer-v4_test':764, 'kev_transfer-v9_development':1264,
 'kev_transfer-v9_test':1264}


def probability_signature(body, prediction):
    return hashlib.sha256(request_digest(body).encode()+b'\n'+canonical(prediction)).hexdigest()


def take_response(pool, body, prediction, kind, converter, model, config):
    """The actual native envelope defines kind; Kev can also name its qid decision."""
    key=request_digest(body)
    for i,response in enumerate(pool[key]):
        converted=json.loads(converter(response,body,model,config,kind,False))
        if canonical(converted)==canonical(prediction):
            return pool[key].pop(i)
    raise ValueError('prediction lacks actual served request/probability witness')


def calibration(binary):
    """Predeclared 10 equal-width ECE; intercept+slope MLE on clipped true logits."""
    if not binary:
        return dict(n=0, ece=None, slope=None, fit_status='NO_BINARY_QUESTIONS')
    bins = [[] for _ in range(10)]
    for p, y in binary:
        bins[min(9, int(p*10))].append((p,y))
    ece = sum(abs(sum(p for p,_ in b)-sum(y for _,y in b)) for b in bins)/len(binary)
    if len({y for _, y in binary}) < 2:
        return dict(n=len(binary), ece=ece, slope=None, fit_status='ONE_CLASS_UNIDENTIFIABLE')
    xs = [math.log(max(1e-12,min(1-1e-12,p))/(1-max(1e-12,min(1-1e-12,p)))) for p,_ in binary]
    a,b = 0.,1.
    for iteration in range(100):
        g0=g1=h00=h01=h11=0.
        for x, (_, y) in zip(xs,binary):
            z=max(-40.,min(40.,a+b*x)); p=1/(1+math.exp(-z)); w=p*(1-p)
            g0+=p-y; g1+=(p-y)*x; h00+=w; h01+=w*x; h11+=w*x*x
        det=h00*h11-h01*h01
        if det <= 1e-12:
            return dict(n=len(binary),ece=ece,slope=None,fit_status='SINGULAR_OR_SEPARATED')
        da=(h11*g0-h01*g1)/det; db=(h00*g1-h01*g0)/det
        a-=da;b-=db
        if not math.isfinite(a+b) or abs(a)>100 or abs(b)>100:
            return dict(n=len(binary),ece=ece,slope=None,fit_status='SEPARATED_OR_DIVERGED')
        if max(abs(da),abs(db))<1e-8:
            return dict(n=len(binary),ece=ece,slope=b,intercept=a,fit_status='CONVERGED',iterations=iteration+1)
    return dict(n=len(binary),ece=ece,slope=None,fit_status='NOT_CONVERGED_100_STEPS')


def target_key(value, probabilities, qtype):
    if qtype == 'noul':
        if type(value) is bool:
            value='true' if value else 'false'
        if value in ('true','false') and set(probabilities)=={'no','yes'}:
            return 'yes' if value=='true' else 'no'
    key=str(value)
    if key not in probabilities:
        raise ValueError('ground-truth label does not match native probability keys')
    return key


def question_metric(distribution, target, qtype):
    keys=list(distribution)
    ps=list(distribution.values())
    if not ps or any(type(p) not in (int,float) or not math.isfinite(p) or p<0 or p>1 for p in ps) or abs(sum(ps)-1)>1e-5:
        raise ValueError('invalid native distribution')
    gold=keys.index(target); guess=max(range(len(ps)),key=ps.__getitem__)
    result=dict(nll=-math.log(max(1e-12,ps[gold])),
                brier=sum((p-(i==gold))**2 for i,p in enumerate(ps)),accuracy=float(guess==gold))
    if qtype=='noul':
        positive='true' if 'true' in distribution else 'yes'
        result['brier']=(distribution[positive]-(target==positive))**2
    if qtype=='score':
        if len(ps)<2 or keys!=[str(i) for i in range(len(ps))]:
            raise ValueError('ordinal score needs its actual ordered numeric levels 0..K-1')
        cumulative=0.;rps=0.
        for i,p in enumerate(ps[:-1]):
            cumulative+=p;rps+=(cumulative-(gold<=i))**2
        expected_grade=sum(i*p for i,p in enumerate(ps));error=expected_grade-gold
        result.update(expected_grade_mae=abs(error),expected_grade_mse=error**2,argmax_grade_mae=abs(guess-gold),rps=rps/(len(ps)-1))
    return result


def main():
    worker_alarm()
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True)
    ap.add_argument('--work',required=True);ap.add_argument('--services',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    plan=read(a.plan);sys.path.insert(0,plan['code_tree'])
    from runtime.bundle_contract import native_prediction
    work=Path(a.work);result=read(work/'completion.json')
    if result['complete'] is not True or set(result['results'])!=set(JK_COUNTS):
        raise ValueError('missing seven native JK populations')
    service_by_signature=defaultdict(list)
    response_files={}
    for path in sorted(Path(a.services).glob('replica*/requests.jsonl')):
        response_files[str(path)]=sha(path)
        with path.open() as f:
            for line in f:
                event=json.loads(line);body=event['request'];response=event['response']
                service_by_signature[request_digest(body)].append(response)
    output=Path(a.output);output.mkdir(exist_ok=False)
    ordered_requests,ordered_tokens=hashlib.sha256(),hashlib.sha256()
    request_counts,token_counts=Counter(),Counter()
    qmetrics=defaultdict(list);binary=[];populations={}
    with (output/'ordered_predictions.jsonl').open('x') as out:
        for name,n in JK_COUNTS.items():
            selection=read(Path(plan['selection_root'])/(name+'.selection.json'))
            ids=selection['selected_ids'];selected=set(ids)
            if len(ids)!=len(selected) or len(selected)!=n:
                raise ValueError('fixed JK selection differs '+name)
            candidates=[p for p in work.rglob(name+'.json') if 'predictions' in p.parts]
            if len(candidates)!=1:
                raise ValueError('missing/ambiguous full predictions '+name)
            predictions=read(candidates[0]);local_ids=[];native_order=hashlib.sha256()
            if set(predictions)!=set(map(str,range(n))):
                raise ValueError('native ordinal prediction indices incomplete '+name)
            for ordinal in range(n):
                row=predictions[str(ordinal)]
                envelope=json.loads(row['origin_prompt']);prediction=json.loads(row['prediction']);gold=json.loads(row['gold'])
                identifier=gold['id'] if name=='jevbench_public' else gold['_meta']['id']
                local_ids.append(identifier)
                body=dict(envelope['request'],model=plan['run_id'])
                response=take_response(service_by_signature,body,prediction,envelope['kind'],
                    native_prediction,plan['run_id'],plan['head_config'])
                key=request_digest(body);tk=request_token_digest(body,response['input_tokens'])
                request_counts[key]+=1;token_counts[tk]+=1
                ordered_requests.update(key.encode()+b'\n');ordered_tokens.update(tk.encode()+b'\n')
                native_order.update(canonical(dict(id=identifier,request_sha256=key))+b'\n')
                if name=='jevbench_public':
                    q=gold['question'];questions=[('decision',q,gold['expected'],prediction['probabilities'])]
                else:
                    questions=[(qid,q,q['label'],prediction['probabilities'][qid]) for qid,q in gold['questions'].items()]
                for qid,q,label,distribution in questions:
                    target=target_key(label,distribution,q['type'])
                    metric=question_metric(distribution,target,q['type']);qmetrics[q['type']].append(metric)
                    if q['type']=='noul':
                        pos='true' if 'true' in distribution else 'yes'
                        binary.append((distribution[pos],int(target==pos)))
                out.write(canonical(dict(population=name,ordinal=ordinal,id=identifier,
                    request=body,response=response,prediction=prediction,gold=gold)).decode()+'\n')
            if len(set(local_ids))!=n or set(local_ids)!=selected:
                raise ValueError('native complete selected population differs '+name)
            populations[name]=dict(records=n,predictions_sha256=sha(candidates[0]),
                result=result['results'][name],native_order_identity_sha256=native_order.hexdigest())
    if any(service_by_signature.values()):
        raise ValueError('unexpected/duplicate served JK requests')
    groups={q:dict(n=len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}) for q,rows in qmetrics.items()}
    if 'score' in groups:
        groups['score']['expected_grade_rmse']=math.sqrt(groups['score'].pop('expected_grade_mse'))
    durable(output/'summary.json',dict(complete=True,records=sum(JK_COUNTS.values()),populations=populations,
        ordered_request_sha256=ordered_requests.hexdigest(),ordered_request_encoder_token_sha256=ordered_tokens.hexdigest(),
        request_occurrence_sha256=counter_digest(request_counts),request_encoder_tokens_occurrence_sha256=counter_digest(token_counts),
        full_ordered_probabilities_sha256=sha(output/'ordered_predictions.jsonl'),
        served_request_files=response_files,qtype_metrics=groups,noul_calibration=calibration(binary),
        metric_rule='Question-weighted native probabilities; NLL clip1e-12, binary Noul Brier(p_true-y)^2, multiclass Choice/Score Brier sum, expected-grade MAE/RMSE and RPS/(K-1), binary ECE10 equal bins and intercept+slope MLE100 Newton steps; no new inference'))


if __name__=='__main__':
    main()
