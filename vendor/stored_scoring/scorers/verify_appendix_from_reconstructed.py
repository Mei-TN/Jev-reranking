"""Read-only, standard-library appendix verification from reconstructed task scores.

main(results, experiments, csv_directory) returns a JSON-serializable report.
For exact cohort scientific-ID matching, attach manifest['models'] as
experiments['_model_registry']. Historical rows outside that registry remain
explicitly outside the raw-linked coverage. No CSV or manuscript is changed.
"""
import csv
import math
import statistics
from pathlib import Path

SOURCES = {'ARC':['arc_easy','arc_challenge'], 'GSM8K':['gsm8k_mc4','gsm8k_mc10'],
           'GPQA':['gpqa','gpqa_fewshot'], 'MMLU':['mmlu','mmlu_fewshot'],
           'HellaSwag':['hellaswag'], 'WinoGrande':['winogrande'], 'chess':['chess']}
MC = [t for ts in SOURCES.values() for t in ts]
KT = ['kev_decision-v7_test','kev_transfer-v4_test','kev_transfer-v9_test']
KD = [t.replace('_test','_development') for t in KT]
SETTINGS = ['F2s44','F2s45','F3s44','F3s45']
BLOCKS = ['E123','E4a','E5','E8','E11']
SIGN_ZERO_TOLERANCE_PP = 1e-9


def sign_counts(xs):
    """Ignore floating-point cancellation below the scored effect resolution."""
    return {'positive':sum(x>SIGN_ZERO_TOLERANCE_PP for x in xs),
            'zero':sum(abs(x)<=SIGN_ZERO_TOLERANCE_PP for x in xs),
            'negative':sum(x< -SIGN_ZERO_TOLERANCE_PP for x in xs)}

def mean(xs):
    return math.fsum(xs)/len(xs)

def views(t):
    f = {s:mean([t[x] for x in ts]) for s,ts in SOURCES.items()}
    return {'MC11':mean([t[x] for x in MC]), 'MC7_equal_source':mean(list(f.values())),
            'Kev6':mean([t[x] for x in KT+KD]), 'Kev_test3':mean([t[x] for x in KT]),
            'Kev_development3':mean([t[x] for x in KD]), 'JevBench':t['jevbench_public'],
            **{'source_'+s:v for s,v in f.items()},
            **{'LOSO_equal_remaining_configurations_'+s:mean([t[x] for x in MC if x not in ts]) for s,ts in SOURCES.items()},
            **{'LOSO_equal_remaining_sources_'+s:mean([v for other,v in f.items() if other!=s]) for s in f}}

def main(core_results, core_experiments, csv_directory):
    q = Path(csv_directory)
    results = core_results.get('results', core_results)
    comparisons = 0
    max_error = 0.0
    failures = []
    file_rows = {}

    def check(a,b,label):
        nonlocal comparisons,max_error
        comparisons += 1
        a,b = float(a),float(b)
        err = abs(a-b)
        if not math.isfinite(a) or not math.isfinite(b) or err>1e-9:
            failures.append({'check':label,'actual':a,'expected':b})
        max_error = max(max_error,err)

    def require(ok,label):
        nonlocal comparisons
        comparisons += 1
        if not ok: failures.append({'check':label,'structural_failure':True})

    def read(name):
        with (q/name).open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        file_rows[name]=len(rows)
        return rows

    coefficients={b:{c:dict(k) for c,k in core_experiments[b]['contrast_coefficients'].items()} for b in BLOCKS}
    coefficients['E11'].update(pointer_two_minus_one={'POINTER_2EP':1,'POINTER_1EP':-1},
                               pool_two_minus_one={'POOL_2EP':1,'POOL_1EP':-1})
    tasks={}; calc={}; abs_tasks={}
    for b in BLOCKS:
        settings={f"{s['fold']}s{s['seed']}":s for s in core_experiments[b]['settings']}
        require(set(settings)==set(SETTINGS),b+'/four settings')
        for setting in SETTINGS:
            abs_tasks[b,setting]={a:results[mid]['tasks_percent'] for a,mid in settings[setting]['models'].items()}
        for cid,coefs in coefficients[b].items():
            for setting in SETTINGS:
                arms=abs_tasks[b,setting]
                t={task:math.fsum(w*arms[a][task] for a,w in coefs.items()) for task in arms[next(iter(coefs))]}
                tasks[b,cid,setting]=t
                calc[b,cid,setting]=views(t)
    require(sum(map(len,coefficients.values()))==23,'23 contrasts')

    for filename,is_loso in [('appendix_paired_contrasts.csv',False),('appendix_loso_complete.csv',True)]:
        rows=read(filename)
        expected={(b,c,v) for b in BLOCKS for c in coefficients[b] for v in calc[b,c,SETTINGS[0]] if (not is_loso or v.startswith('LOSO_'))}
        actual={(r['block'],r['contrast_id'],r['view']) for r in rows}
        require(actual==expected and len(actual)==len(rows),filename+'/unique full coverage')
        for r in rows:
            key=(r['block'],r['contrast_id']); view=r['view']
            xs=[calc[*key,s][view] for s in SETTINGS]
            label='/'.join((*key,view))
            for s,x in zip(SETTINGS,xs):check(r[s],x,label+'/'+s)
            for field,val in {'mean_pp':mean(xs),'descriptive_se_pp':statistics.stdev(xs)/2,
                              'minimum_pp':min(xs),'maximum_pp':max(xs),
                              **sign_counts(xs)}.items():check(r[field],val,label+'/'+field)
            if view.startswith('LOSO_'):
                for prefix,rule in [('LOSO_equal_remaining_configurations_','equal_remaining_configurations'),('LOSO_equal_remaining_sources_','equal_remaining_sources')]:
                    if view.startswith(prefix):
                        require(r['omitted_source']==view.removeprefix(prefix) and r['loso_weighting']==rule,label+'/omission rule')

    rows=read('appendix_contrast_coefficients.csv')
    expected={(b,c,a) for b in BLOCKS for c,k in coefficients[b].items() for a in k}
    require({(r['block'],r['contrast_id'],r['arm']) for r in rows}==expected and len(rows)==len(expected),'coefficient coverage')
    for r in rows:check(r['coefficient'],coefficients[r['block']][r['contrast_id']][r['arm']],'coefficient')
    rows=read('appendix_source_mapping.csv')
    require({(r['source'],r['configuration'],r['same_question_pair']) for r in rows}==
            {(s,t,str(s in ['GSM8K','GPQA','MMLU'])) for s,ts in SOURCES.items() for t in ts} and len(rows)==11,'source mapping and question-pair identity')
    rows=read('appendix_budget_task_settings.csv')
    expected={(c,t,s) for c in coefficients['E11'] for t in MC for s in SETTINGS}
    require({(r['contrast_id'],r['task'],r['setting']) for r in rows}==expected and len(rows)==220,'budget task setting coverage')
    for r in rows:
        x=tasks['E11',r['contrast_id'],r['setting']][r['task']]
        check(r['task_change_pp'],x,'budget task effect');check(r['MC11_contribution_pp'],x/11,'budget contribution')
        require(r['task'] in SOURCES[r['source']],'budget source identity')
    rows=read('appendix_budget_task_means.csv')
    require({r['task'] for r in rows}==set(MC) and len(rows)==11,'budget mean coverage')
    for r in rows:
        t=r['task'];require(t in SOURCES[r['source']],'budget mean source identity')
        for a in ['POINTER_1EP','POOL_1EP','POINTER_2EP','POOL_2EP']:
            check(r[a],mean([abs_tasks['E11',s][a][t] for s in SETTINGS]),'budget mean absolute')
        for c in coefficients['E11']:
            x=mean([tasks['E11',c,s][t] for s in SETTINGS])
            check(r[c],x,'budget mean effect');check(r[c+'_MC11_contribution'],x/11,'budget mean contribution')
    rows=read('appendix_budget_absolute_settings.csv')
    require({(r['setting'],r['task']) for r in rows}=={(s,t) for s in SETTINGS for t in MC} and len(rows)==44,'budget absolute coverage')
    for r in rows:
        for a in ['POINTER_1EP','POOL_1EP','POINTER_2EP','POOL_2EP']:
            check(r[a],abs_tasks['E11',r['setting']][a][r['task']],'budget absolute')

    rows=read('appendix_cohort_models.csv')
    registry=core_experiments.get('_model_registry',[])
    identities={m['scientific_model_id']:m for m in registry}
    cohort_verified=[];cohort_uncovered=[]
    require(len(rows)==48,'cohort ledger row count')
    for r in rows:
        if r['model_id'] not in identities:
            cohort_uncovered.append(r['model_id']);continue
        m=identities[r['model_id']]
        require(r['setting']==f"{m['fold']}s{m['seed']}",'cohort exact identity and setting')
        v=views(results[m['id']]['tasks_percent'])
        for col,key in [('JevBench','JevBench'),('Kev','Kev6'),('MC11','MC11')]:check(r[col],v[key],'cohort '+r['model_id']+'/'+col)
        cohort_verified.append(r['model_id'])
    if registry:
        require(len(cohort_verified)==16 and len(cohort_uncovered)==32,'16 current / 32 historical cohort coverage')
    return {'schema':'jev-appendix-raw-linked-verification-v1','status':'PASS' if not failures else 'FAIL',
            'numeric_and_structural_comparisons':comparisons,'max_absolute_numeric_difference':max_error,
            'failures':failures,'file_rows':file_rows,'paired_contrasts':23,'views_per_contrast':27,
            'loso_rows':322,'loso_setting_values':1288,'core_model_count':len(results),
            'cohort_current_rows_verified':len(cohort_verified),'cohort_current_model_ids':cohort_verified,
            'cohort_historical_rows_outside_core_coverage':len(cohort_uncovered),
            'cohort_historical_model_ids_outside_core_coverage':cohort_uncovered,
            'coverage':'23 paired contrasts and all aggregation views, both LOSO rules, source/coefficients, and all E11 numerical CSV values link to reconstructed current-model task scores. Exact current cohort endpoint linkage requires _model_registry. Historical cohort scores, hardware and initialization prose are outside this probability-derived verification. Qwen is not represented by these appendix CSVs.',
            'csv_files_modified':False,'original_verification_layer_preserved':False,
            'sign_count_rule':'Compare stored setting values to reconstructed values within 1e-9 pp. Compute positive/zero/negative counts from the reconstructed values, treating abs(x)<=1e-9 pp as zero. This tolerance applies only to sign counts; stored values, means, SEs, and extrema remain unchanged.',
            'sign_count_zero_tolerance_pp':SIGN_ZERO_TOLERANCE_PP,'comparison_tolerance_pp':1e-9}

if __name__=='__main__':
    import argparse,json,zipfile
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--archive',required=True)
    p.add_argument('--csv-directory',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    results=json.loads(Path(args.results).read_text(encoding='utf-8'))
    with zipfile.ZipFile(args.archive) as z:
        experiments=json.loads(z.read('experiments.json'))
        experiments['_model_registry']=json.loads(z.read('manifest.json'))['models']
    report=main(results,experiments,args.csv_directory)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','numeric_and_structural_comparisons','max_absolute_numeric_difference','cohort_current_rows_verified']}))
    raise SystemExit(0 if report['status']=='PASS' else 1)
