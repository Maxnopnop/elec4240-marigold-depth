import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(snapshot_download('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
        revision='8078d68a9c75a972131914f6afd0c1723be0da7f',local_dir=str(Path(a.root)/'expert'),
        allow_patterns=['*.json','*.safetensors']))
