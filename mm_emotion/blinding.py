"""Export shuffled generation texts and a separate private condition key."""
import argparse
import csv
import json
import random
from pathlib import Path


def export_blinded(payload, output, seed=42):
    rows=payload.get('results',[])
    if not rows or any(r.get('origin')!='real_model_generation' for r in rows):
        raise ValueError('Expected recorded model generations')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    shuffled=list(rows);random.Random(seed).shuffle(shuffled)
    key=[]
    with (output/'annotations.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_id','prompt','text','emotion_rating','helpfulness_rating','notes'])
        writer.writeheader()
        for i,row in enumerate(shuffled,1):
            sample_id=f'sample-{i:05d}'
            writer.writerow({'sample_id':sample_id,'prompt':row['prompt'],'text':row['output']['text']})
            key.append({'sample_id':sample_id,**{k:v for k,v in row.items() if k!='output'}})
    (output/'condition_key_PRIVATE.json').write_text(json.dumps(key,indent=2),encoding='utf8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    print(export_blinded(json.loads(Path(args.run).read_text(encoding='utf8')),args.out,args.seed))
