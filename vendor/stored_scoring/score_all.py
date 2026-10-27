"""Fresh-directory, standard-library-only verification of the scoring package."""
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'scorers'))
from score_compact_records import main as score_core
from score_auxiliary_records import main as score_auxiliary
from check_score_boundaries import check as check_boundaries
from verify_appendix_from_reconstructed import main as verify_appendix

def main():
    # Scoring consumes only population definitions, probabilities and gold.
    # Frozen aggregate targets are loaded only after all scores are computed.
    results={name:scorer(HERE/'modules'/file) for name,scorer,file in [
        ('core56',score_core,'core56_stored_score_supplement.zip'),
        ('qwen8',score_core,'qwen8_stored_score_supplement.zip'),
        ('auxiliary',score_auxiliary,'auxiliary_stored_score_supplement.zip')]}
    boundaries=check_boundaries(HERE/'modules/core56_stored_score_supplement.zip')
    with zipfile.ZipFile(HERE/'modules/core56_stored_score_supplement.zip') as core:
        experiments=json.loads(core.read('experiments.json'))
        experiments['_model_registry']=json.loads(core.read('manifest.json'))['models']
    appendix=verify_appendix(results['core56'],experiments,HERE/'summary_views')
    if appendix['status']!='PASS':raise ValueError('appendix reconstruction mismatch: '+json.dumps(appendix['failures']))
    expected=json.loads((HERE/'expected_aggregates.json').read_text());differences=[]
    def compare(actual,target,path):
        if isinstance(target,dict):
            for k,v in target.items():compare(actual[k],v,path+'/'+k)
        elif isinstance(target,list):
            if len(actual)!=len(target):raise ValueError('length mismatch '+path)
            for i,v in enumerate(target):compare(actual[i],v,path+'/'+str(i))
        elif isinstance(target,(int,float)):
            d=abs(actual-target)
            if not math.isfinite(actual) or d>1e-9:raise ValueError('aggregate mismatch '+path)
            differences.append(d)
        elif actual!=target:raise ValueError('identity mismatch '+path)
    for module in results:compare(results[module],expected[module],module)
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,pin in manifest['files'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('package member changed '+name)
    registry=json.loads((HERE/'model_registry.json').read_text())
    executions=json.loads((HERE/'execution_registry.json').read_text())
    model_count=len({m['model_id'] for m in registry})
    core_tasks=sum(results[k]['tasks'] for k in ['core56','qwen8'])
    auxiliary_conditions=sum(len(m['conditions']) for m in results['auxiliary']['results'].values())
    eligible=sum(v[1] for k in ['core56','qwen8'] for m in results[k]['results'].values() for v in m['correct_denominator'].values())+results['auxiliary']['scored_question_occurrences']
    assert model_count==72 and len(executions)==96 and core_tasks==1152 and auxiliary_conditions==200 and eligible==3722928
    output=HERE/'verification_outputs';output.mkdir(exist_ok=True)
    for module,result in results.items():(output/(module+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    (output/'appendix.json').write_text(json.dumps(appendix,indent=2)+'\n')
    report=dict(status='PASS',fresh_directory_verification=True,modules=['core56','qwen8','auxiliary'],
        independent_checkpoint_models=model_count,execution_modules=len(executions),core_model_tasks=core_tasks,auxiliary_model_conditions=auxiliary_conditions,
        eligible_question_occurrences=eligible,integer_differences=0,
        aggregate_comparisons=len(differences),maximum_aggregate_difference_pp=max(differences),boundaries=boundaries,
        appendix_comparisons=appendix['numeric_and_structural_comparisons'],maximum_appendix_difference_pp=appendix['max_absolute_numeric_difference'],
        coverage=manifest['coverage'],scope='Offline stored-output scoring only; no model, GPU, service, internet, native fork or training reproduction.')
    (output/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
