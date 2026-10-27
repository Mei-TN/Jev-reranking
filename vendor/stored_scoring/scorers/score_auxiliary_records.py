"""Score stored SocialIQA, PIQA and disjoint-GPQA candidate values offline."""
from array import array
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import zipfile

def require(ok,message):
    if not ok:raise ValueError(message)

def stats(values):
    mean=math.fsum(values)/len(values)
    sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
    return dict(n_settings=len(values),mean=mean,sample_sd=sd,descriptive_se=sd/math.sqrt(len(values)),per_setting=values)

def main(path):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('manifest.json'));out={};groups=defaultdict(dict)
        for module in manifest['modules']:
            population=json.loads(z.read(module+'/population.json'))
            identities=[(r['condition'],r['source_ordinal'],r['item_id']) for r in population]
            require(len(set(identities))==len(identities),'duplicate occurrence identity')
            for m in manifest['modules'][module]['models']:
                values=array('d');values.frombytes(z.read(module+'/'+m['id']+'.f64le'))
                if sys.byteorder!='little':values.byteswap()
                cursor=0;conditions=defaultdict(lambda:dict(n=0,correct=0,ties=0,uniform_tie_credit=0.0,correct_bits='',prediction_indices=[]))
                for row in population:
                    n=len(row['candidate_order']);v=values[cursor:cursor+n];cursor+=n
                    require(len(v)==n and all(math.isfinite(x) and 0<=x<=1 for x in v),'invalid candidate values')
                    guess=max(range(n),key=v.__getitem__);mx=v[guess]
                    maxima=[i for i,x in enumerate(v) if x==mx]
                    correct=int(row['candidate_order'][guess]==row['gold'])
                    credit=sum(row['candidate_order'][i]==row['gold'] for i in maxima)/len(maxima)
                    c=conditions[row['condition']];c['n']+=1;c['correct']+=correct;c['ties']+=len(maxima)>1
                    c['uniform_tie_credit']+=credit;c['correct_bits']+=str(correct);c['prediction_indices'].append(guess)
                require(cursor==len(values),'unused candidate values')
                native=json.loads(z.read(module+'/'+m['id']+'.reference.json'))
                for condition,c in conditions.items():
                    ref=native[condition]
                    require(c['n']==ref['n'] and c['correct']==ref['correct'] and c['ties']==ref['ties'],'native integer mismatch '+m['id']+'/'+condition)
                    require(c['correct_bits']==ref['correct_bits'] and c['prediction_indices']==ref['prediction_indices'],'native per-item mismatch '+m['id']+'/'+condition)
                    c['accuracy']=c['correct']/c['n'];c['accuracy_percent']=100*c['accuracy']
                    c['uniform_tie_accuracy']=c.pop('uniform_tie_credit')/c['n']
                out[m['id']]=dict(module=module,fold=m['fold'],seed=m['seed'],arm=m['arm'],epochs=m.get('epochs'),conditions=dict(conditions))
                cohort=module+('_epoch'+str(m['epochs']) if module=='socialiqa' else '')
                groups[cohort].setdefault((m['fold'],m['seed']),{})[m['arm']]=m['id']
        derived={}
        for cohort,settings in groups.items():
            settings=[dict(fold=f,seed=s,models=m) for (f,s),m in sorted(settings.items())]
            require(len(settings)==4 and all(set(s['models'])=={'pointer','pooling'} for s in settings),'incomplete setting pairs')
            condition_names=sorted(out[settings[0]['models']['pointer']]['conditions'])
            summary={}
            for condition in condition_names:
                vectors={arm:[out[s['models'][arm]]['conditions'][condition]['accuracy_percent'] for s in settings] for arm in ['pointer','pooling']}
                delta=[b-a for a,b in zip(vectors['pointer'],vectors['pooling'])]
                summary[condition]={arm:stats(values) for arm,values in vectors.items()}
                summary[condition]['pooling_minus_pointer_pp']=stats(delta)
            entry=dict(settings=settings,conditions=summary)
            if cohort.startswith('socialiqa'):
                entry['interaction_nli_minus_direct_pp']=stats([b-a for a,b in zip(summary['direct']['pooling_minus_pointer_pp']['per_setting'],summary['nli']['pooling_minus_pointer_pp']['per_setting'])])
            if cohort=='fixed_interfaces':
                names=[c for c in condition_names if c.startswith('piqa_nli_')]
                require(len(names)==18,'expected all 18 PIQA templates')
                entry['template_effect_range_pp']=[min(summary[c]['pooling_minus_pointer_pp']['mean'] for c in names),max(summary[c]['pooling_minus_pointer_pp']['mean'] for c in names)]
                entry['minimum_setting_template_delta_pp']=min(x for c in names for x in summary[c]['pooling_minus_pointer_pp']['per_setting'])
                entry['nli_P0D0_minus_direct_interaction_pp']=stats([b-a for a,b in zip(summary['piqa_direct']['pooling_minus_pointer_pp']['per_setting'],summary['piqa_nli_P0D0']['pooling_minus_pointer_pp']['per_setting'])])
                sensitivity={}
                for group in ['D0','D1','D2','all_18']:
                    templates=[c for c in names if group=='all_18' or c.endswith(group)]
                    rows=[]
                    for i,setting in enumerate(settings):
                        ptr=[summary[c]['pointer']['per_setting'][i] for c in templates]
                        pool=[summary[c]['pooling']['per_setting'][i] for c in templates]
                        delta=[b-a for a,b in zip(ptr,pool)]
                        rows.append(dict(fold=setting['fold'],seed=setting['seed'],templates=templates,n_templates=len(templates),
                            pointer_mean=math.fsum(ptr)/len(ptr)/100,pooling_mean=math.fsum(pool)/len(pool)/100,
                            pointer_template_sd_pp=stats(ptr)['sample_sd'],pooling_template_sd_pp=stats(pool)['sample_sd'],
                            mean_paired_delta_pp=math.fsum(delta)/len(delta),paired_delta_template_sd_pp=stats(delta)['sample_sd'],
                            worst_paired_delta_pp=min(delta),paired_deltas_by_template_pp=dict(zip(templates,delta))))
                    sensitivity[group]=dict(settings=rows,
                        mean_template_paired_delta_pp_across_settings=stats([r['mean_paired_delta_pp'] for r in rows]),
                        settingwise_worst_paired_delta_pp=stats([r['worst_paired_delta_pp'] for r in rows]),
                        worst_template_of_four_setting_mean_paired_delta_pp=min(summary[c]['pooling_minus_pointer_pp']['mean'] for c in templates))
                entry['template_sensitivity']=sensitivity
            derived[cohort]=entry
        return dict(status='PASS',model_modules=len(out),scored_question_occurrences=sum(c['n'] for m in out.values() for c in m['conditions'].values()),results=out,derived=derived,
                    scope=manifest['scope'],statistical_units='Four specified overlapping fold/seed settings; SE is descriptive. Templates and questions are not independent training replicates.')

if __name__=='__main__':
    result=main(sys.argv[1]);target=Path(sys.argv[2]) if len(sys.argv)>2 else Path('auxiliary_reconstructed_scores.json')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],model_modules=result['model_modules'],question_occurrences=result['scored_question_occurrences'],output=str(target))))
