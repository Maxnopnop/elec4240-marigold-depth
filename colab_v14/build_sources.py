"""Derive an isolated, reviewable V14 implementation; never modifies V13."""
from pathlib import Path
P=Path(__file__).parent
OLD=P.parent/'colab_v13'
def put(n,s): (P/n).write_text(s,encoding='utf-8')
s=(OLD/'prepare_scale_cache.py').read_text().split('def main():')[0]
s=s.replace("Path('/content/v13_cloud_scale')","Path('/content/v14_pretraining_lora')")
s=s.replace("DRIVE = Path(","OLD_DRIVE = Path(")
s=s.replace("EXPECTED_MANIFEST =", "DRIVE = OLD_DRIVE / 'pretraining_lora_v14'\nEXPECTED_MANIFEST =")
put('io_utils.py',s)
s=(OLD/'scale_runtime.py').read_text().replace('from prepare_scale_cache import','from io_utils import')
s=s.replace('EXPECTED_MANIFEST\n','EXPECTED_MANIFEST, OLD_DRIVE\n')
s=s.replace("WORK = DRIVE / 'scale_experiment'","WORK = DRIVE")
s=s.replace("['pixel', 'pair_always', 'pair_ready']","['A_random', 'B_generative', 'C_depth']").replace('[17, 29]','[17, 29, 43]').replace('[2048, 4096]','[512, 2048]').replace('STEPS = 4096','STEPS = 2048')
s=s.replace("assert sha(ROOT/'cloud_data_manifest.json') == EXPECTED_MANIFEST","assert sha(ROOT/'cloud_data_manifest.json') == read(DRIVE/'data_receipt.json')['manifest_sha256']")
start=s.index('def setup(');end=s.index('\ndef text_prompt',start)
s=s[:start]+'''def setup(seed, checkpoint=None, arm='C_depth'):
    from model_setup import make_model
    return make_model(arm,seed,checkpoint)

'''+s[end:]
s=s.replace('len(result) == 4096','len(result) == 2048')
s=s.replace("read(DRIVE/'scale_cache/cache_audit.json')","read(OLD_DRIVE/'scale_cache/cache_audit.json')").replace("sha(DRIVE/'scale_cache'/name)","sha(OLD_DRIVE/'scale_cache'/name)").replace("copy_verified(DRIVE/'scale_cache'/name, local)","copy_verified(OLD_DRIVE/'scale_cache'/name, local)")
s=s.replace("cache = {'rgb': {}, 'mask': {}, 'text': {}}", "cache = {'rgb': {}, 'mask': {}, 'text': {}}\n    rr=[r for r in rows() if r['split']=='train']\n    allowed={'rgb':{r['image_id'] for r in rr},'mask':{r['ann_id'] for r in rr},'text':{r['sent_id'] for r in rr}}")
s=s.replace("cache[k].update(shard[k])","cache[k].update({i:v for i,v in shard[k].items() if i in allowed[k]})")
s=s.replace("len(cache['rgb']) == 4096 and len(cache['mask']) == 8192","len(cache['rgb']) == 2048 and len(cache['mask']) == 4096")
s=s.replace("cache, targets, pair, arm, seed, step)","cache, targets, pair, 'pixel', seed, step)")
s=s.replace('for step in [2048,4096]:','for step in [1024,2048]:')
put('runtime.py',s)
s=(OLD/'scale_train.py').read_text().replace('from scale_runtime import','from runtime import')
s=s.replace("else 'large_train_images'", "else 'large_train_images'").replace('if size == 2048','if size == 512')
s=s.replace("setup(seed, ROOT/f'initial_seed{seed}.pt')","setup(seed, arm=arm)")
s=s.replace('schedule(plan[\'small_train_images\' if size == 512 else \'large_train_images\'])','schedule(plan[\'small_train_images\' if size == 512 else \'large_train_images\'],STEPS)')
s=s.replace('[2048,4096]','[1024,2048]').replace("assert len(set(r['image_id'] for r in history[:2048])) == 2048", "assert len(set(r['image_id'] for r in history[:1024])) == min(size,1024)")
s=s.replace('Matched finite 12-run','Matched finite 18-run').replace("budget.reserve(600, name+':setup')","budget.reserve(900, name+':setup')")
put('train.py',s)
s=(OLD/'scale_evaluate.py').read_text().replace('from scale_runtime import','from runtime import').replace('setup(seed,checkpoint)','setup(seed,checkpoint,arm=arm)')
put('evaluate.py',s)
s=(OLD/'scale_preflight.py').read_text().replace('from scale_runtime import','from runtime import')
s=s.replace("DRIVE/'scale_preflight'","DRIVE/'preflight'").replace("ROOT/'scale_runtime.py'","ROOT/'runtime.py'").replace("setup(17, ROOT/'initial_seed17.pt')","setup(17, arm=arm)")
s=s.replace("'scale_runtime.py','scale_train.py','scale_evaluate.py','scale_report.py','scale_preflight.py','scale_run.py','scale_queue.py','frozen_functions.py','prepare_scale_cache.py'","'runtime.py','train.py','evaluate.py','report.py','preflight.py','run.py','model_setup.py','prepare.py','frozen_functions.py','io_utils.py'")
s=s.replace('budget.reserve(660','budget.reserve(900').replace('timeout=600','timeout=840')
s=s.replace("result={'row':", "result={'initialization_audit':pipe.initialization_audit,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'row':")
s=s.replace("print('PRODUCTION PREFLIGHT PASSED',flush=True)","assert len({read(dest/a/'parent.json')['initialization_audit']['lora_sha256'] for a in ARMS})==1\n    print('PRODUCTION PREFLIGHT PASSED',flush=True)")
put('preflight.py',s)
s=(OLD/'scale_report.py').read_text().replace('from scale_runtime import','from runtime import')
start=s.index('def contrasts():');end=s.index('\ndef holm',start)
s=s[:start]+'''def contrasts():
    # Four primary initialization contrasts plus three data-budget contrasts.
    return [((a,n,2048),(b,n,2048)) for n in SIZES for a,b in [('B_generative','A_random'),('C_depth','B_generative')]] + [((a,2048,2048),(a,512,2048)) for a in ARMS]

'''+s[end:]
s=s.replace('[2048,4096]','[1024,2048]').replace('len(tests)==26','len(tests)==14')
start=s.index('    descriptive=[]');end=s.index("    write(OUT/'analysis.json'",start)
s=s[:start]+"    descriptive=[]\n"+s[end:]
s=s.replace('two seeds','three seeds').replace('all26','all14').replace("'contrasts':26","'contrasts':14").replace('V13 scaling study','V14 LoRA initialization study').replace('data and update scaling','backbone initialization under matched LoRA adaptation')
s=s.replace("caution='Exploratory", "caution='Shared pretrained VAE/text encoder retained even in A; no from-scratch, convergence or full-parameter claim. B uses pinned community SD2 mirror, not independently authenticated original bytes. C differs in depth adaptation and training recipe; no isolated objective causality. Pretraining costs excluded from downstream accounting. Exploratory")
put('report.py',s)
