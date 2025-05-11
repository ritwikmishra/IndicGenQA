

# calculates the times first is performed better than second id
# t40_v2

import argparse, glob, json
from collections import Counter

import wandb
wandb.login(key=input('enter wandb key:'))

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model") # gn3vgn22 or gn3v22 (based on what was used in t40)
args = parser.parse_args()

run = wandb.init(project="genmqna2_chatgpt", id=args.id, name=args.id, resume='allow')

args_dict = vars(args)
print(json.dumps(args_dict, indent=4))

ml = glob.glob('data/jsonData2/out/'+args.id+'/test_comparison_gpt-4o/*.json')

print('Length of list:', len(ml))

ml2 = []

# typical response looks like this:
# "option2\noption1 is data/jsonData2/out/gn3/test_output_ft/\noption2 is data/jsonData2/out/gn22/test_output_ft/"

for fname in ml:
    with open(fname, 'r') as file:
        data = json.loads(file.read())
    r = data['response'].split('\n')
    if 'option1' in r[0]:
        gn = r[1].split()[-1].split('/')[3] # this means select 2nd line and then extract the gn number
    else:
        gn = r[2].split()[-1].split('/')[3] # this means select 3rd line and then extract the gn number
    ml2.append(gn)

print(Counter(ml2))
    
