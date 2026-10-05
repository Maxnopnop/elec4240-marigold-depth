"""Cloud-only original-data and pinned model preparation, no training."""
from io_utils import *
import random,subprocess
B_ID='sd2-community/stable-diffusion-2'
B_REV='2511124fabf8bf30c0ca9dccd9729e7f0f2fa669'
C_ID='prs-eth/marigold-depth-v1-1'
C_REV='9571e7123e258cf052b4e54241f17971c290e9a8'

def main():
    assert sys.platform=='linux' and Path('/content').is_dir()
    assert (OLD_DRIVE/'PAUSE').exists(),'Old experiment must remain paused'
    from huggingface_hub import snapshot_download,HfApi
    paths={};model_receipts={}
    for arm,repo,rev,patterns in [('B',B_ID,B_REV,['unet/config.json','unet/*fp16.safetensors']),
        ('C',C_ID,C_REV,['model_index.json','unet/config.json','unet/*fp16.safetensors','vae/config.json','vae/*fp16.safetensors','text_encoder/config.json','text_encoder/*fp16.safetensors','tokenizer/*','scheduler/*'])]:
        write(DRIVE/'status.json',{'stage':'model_download','model':repo})
        path=Path(snapshot_download(repo,revision=rev,allow_patterns=patterns,max_workers=2))
        info=HfApi().model_info(repo,revision=rev,files_metadata=True)
        hashes={}
        for f in info.siblings:
            p=path/f.rfilename
            if p.is_file():
                hashes[f.rfilename]=sha(p)
                if f.lfs:assert hashes[f.rfilename]==f.lfs.sha256
        assert any(n.endswith('.safetensors') for n in hashes)
        paths[arm]=str(path);model_receipts[arm]={'repo':repo,'revision':rev,'sha256':hashes}
    write(ROOT/'model_paths.json',paths);write(DRIVE/'model_receipts.json',model_receipts)
    oldplan=read(ROOT/'scale_plan.json')
    large=sorted(oldplan['small_train_images'] if len(oldplan['small_train_images'])==2048 else oldplan['large_train_images']);assert len(large)==2048
    shuffled=large.copy();random.Random(424114).shuffle(shuffled)
    plan={'small_train_images':sorted(shuffled[:512]),'large_train_images':large,
          'fresh_holdout_images':oldplan['fresh_holdout_images'],'arms':['A_random','B_generative','C_depth'],
          'seeds':[17,29,43],'steps':2048,'checkpoints':[1024,2048]}
    write(ROOT/'scale_plan.json',plan)
    from prepare_migration import main as acquire
    # Selection occurs before loading images or observing any held-out predictions.
    inp=read(ROOT/'migration_inputs.json')
    selected=set(large)|set(plan['fresh_holdout_images'])
    inp['records']=[r for r in inp['records'] if r['image_id'] in selected]
    write(ROOT/'migration_inputs.json',inp)
    write(DRIVE/'status.json',{'stage':'original_data_restore','train_images':2048,'holdout_images':320})
    acquire()
    write(DRIVE/'data_receipt.json',{'manifest_sha256':sha(ROOT/'cloud_data_manifest.json'),
        'plan_sha256':sha(ROOT/'scale_plan.json'),'train_images':2048,'holdout_images':320,'heldout_inference':False})
    for n in ['cloud_data_manifest.json','scale_plan.json','migration_audit.json','migration_inputs.json']:
        copy_verified(ROOT/n,DRIVE/'inputs'/n)
    print('V14_PREPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()
