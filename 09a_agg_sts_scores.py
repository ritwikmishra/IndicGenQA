# this code calculates the aggregate sts scores given an experiment id
# t32

import glob, pytz, json
from tqdm import tqdm
import argparse, datetime
import os, re
import numpy as np

IST = pytz.timezone('Asia/Kolkata') # your time zone here

import wandb
wandb.login(key='0ac8f8215b27765188818fd25f1c77df38ff24c2')

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
args = parser.parse_args()

print('Agregating t32.py',datetime.datetime.now(IST).strftime("%c"))

run = wandb.init(project="genmqna2_sts_agg", id=args.id, name=args.id, resume='allow')

output = "data/jsonData2/out/"+args.id+"/test_sts/*.txt"

ml = glob.glob(output)

id_to_test_file = {
    'gn3': 'data/jsonData2/indie_gen_test_setting0.jsonl',
    'gn5': 'data/jsonData2/indie_gen_test_setting0.jsonl',
    'gn9': 'data/jsonData2/indie_gen_test_setting0.jsonl',
    'gn6': 'data/jsonData2/indie_gen_test_setting1.jsonl',
    'gn7': 'data/jsonData2/indie_gen_test_setting1.jsonl',
    'gn8': 'data/jsonData2/indie_gen_test_setting1.jsonl',
    'gn10': 'data/jsonData2/indie_gen_test_setting2.1.jsonl',
    'gn11': 'data/jsonData2/indie_gen_test_setting2.1.jsonl',
    'gn12': 'data/jsonData2/indie_gen_test_setting2.1.jsonl',
    'gn13': 'data/jsonData2/indie_gen_test_setting2.2.jsonl',
    'gn14': 'data/jsonData2/indie_gen_test_setting2.2.jsonl',
    'gn15': 'data/jsonData2/indie_gen_test_setting2.2.jsonl',
    'gn16': 'data/jsonData2/indie_gen_test_setting3.jsonl',
    'gn17': 'data/jsonData2/indie_gen_test_setting3.jsonl',
    'gn18': 'data/jsonData2/indie_gen_test_setting3.jsonl',
    'gn19': 'data/jsonData2/indie_gen_test_setting4.jsonl',
    'gn20': 'data/jsonData2/indie_gen_test_setting4.jsonl',
    'gn21': 'data/jsonData2/indie_gen_test_setting4.jsonl',
    'gn22': 'data/jsonData2/indie_gen_test_setting5.jsonl',
    'gn23': 'data/jsonData2/indie_gen_test_setting5.jsonl',
    'gn24': 'data/jsonData2/indie_gen_test_setting5.jsonl',
    'gn25': 'data/jsonData2/gen2oie_gen_test_setting1.jsonl',
    'gn26': 'data/jsonData2/gen2oie_gen_test_setting1.jsonl',
    'gn27': 'data/jsonData2/gen2oie_gen_test_setting1.jsonl',
    'gn28': 'data/jsonData2/gen2oie_gen_test_setting2.1.jsonl',
    'gn29': 'data/jsonData2/gen2oie_gen_test_setting2.1.jsonl',
    'gn30': 'data/jsonData2/gen2oie_gen_test_setting2.1.jsonl',
    'gn31': 'data/jsonData2/gen2oie_gen_test_setting3.jsonl',
    'gn32': 'data/jsonData2/gen2oie_gen_test_setting3.jsonl',
    'gn33': 'data/jsonData2/gen2oie_gen_test_setting3.jsonl',
    'gn34': 'data/jsonData2/gen2oie_gen_test_setting4.jsonl',
    'gn35': 'data/jsonData2/gen2oie_gen_test_setting4.jsonl',
    'gn36': 'data/jsonData2/gen2oie_gen_test_setting4.jsonl',
    'gn48': 'data/jsonData2/indie_gen_test_setting6.jsonl',
    'gn49': 'data/jsonData2/indie_gen_test_setting6.jsonl',
    'gn50': 'data/jsonData2/indie_gen_test_setting6.jsonl',
    'gn52': 'data/jsonData2/indie_gen_test_setting5_3.jsonl',
    'gn53': 'data/jsonData2/indie_gen_test_setting5_3.jsonl',
    'gn54': 'data/jsonData2/indie_gen_test_setting5_3.jsonl',
    'gn55': 'data/jsonData2/indie_gen_test_setting5_4.jsonl',
    'gn56': 'data/jsonData2/indie_gen_test_setting5_4.jsonl',
    'gn57': 'data/jsonData2/indie_gen_test_setting5_4.jsonl',
    'gn58': 'data/jsonData2/indie_gen_test_setting5_5.jsonl',
    'gn59': 'data/jsonData2/indie_gen_test_setting5_5.jsonl',
    'gn60': 'data/jsonData2/indie_gen_test_setting5_5.jsonl',
    'gn61': 'data/jsonData2/indie_gen_test_setting5_6.jsonl',
    'gn62': 'data/jsonData2/indie_gen_test_setting5_6.jsonl',
    'gn63': 'data/jsonData2/indie_gen_test_setting5_6.jsonl'
}

agg_d = {"baseVsSilver": [], "ftVsSilver": [], "baseVsFt": []}
agg_langs = {'hindi': {"baseVsSilver": [], "ftVsSilver": [], "baseVsFt": []}, 'tamil': {"baseVsSilver": [], "ftVsSilver": [], "baseVsFt": []}, 'telugu': {"baseVsSilver": [], "ftVsSilver": [], "baseVsFt": []}, 'urdu': {"baseVsSilver": [], "ftVsSilver": [], "baseVsFt": []}}

test_file_path = id_to_test_file[args.id]
# read jsonl file
test_data = []
with open(test_file_path, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines, ncols=100, desc='Reading Test Data'):
        test_data.append(json.loads(line))

for fname in tqdm(ml, ncols=100, desc='collecting'):
    try:
        with open(fname) as f:
            data = json.loads(f.read())
    except Exception as e:
        print(e)
        print(fname)
        print('-'*50,'error in reading file','-'*50)
        continue
    agg_d['baseVsSilver'].append(data['baseVsSilver'])
    agg_d['ftVsSilver'].append(data['ftVsSilver'])
    agg_d['baseVsFt'].append(data['baseVsFt'])

    url = test_data[int(fname.split('/')[-1].replace('.txt',''))]['url']
    lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', url)[0].split('/')[1]
    agg_langs[lang]['baseVsSilver'].append(data['baseVsSilver'])
    agg_langs[lang]['ftVsSilver'].append(data['ftVsSilver'])
    agg_langs[lang]['baseVsFt'].append(data['baseVsFt'])

assert len(agg_d['baseVsSilver']) == len(agg_d['ftVsSilver']) == len(agg_d['baseVsFt'])
assert len(agg_d['baseVsSilver']) == sum([len(agg_langs[x]['baseVsSilver']) for x in agg_langs.keys()])

agg_d['baseVsSilver'] = list(np.array(agg_d['baseVsSilver']).mean(axis=0))
agg_d['ftVsSilver'] = list(np.array(agg_d['ftVsSilver']).mean(axis=0))
agg_d['baseVsFt'] = list(np.array(agg_d['baseVsFt']).mean(axis=0))

agg_d['baseVsSilver'].append(sum(agg_d['baseVsSilver'])/len(agg_d['baseVsSilver']))
agg_d['ftVsSilver'].append(sum(agg_d['ftVsSilver'])/len(agg_d['ftVsSilver']))
agg_d['baseVsFt'].append(sum(agg_d['baseVsFt'])/len(agg_d['baseVsFt']))

print(agg_d)

for lang in agg_langs.keys():
    agg_d_lang = agg_langs[lang]
    agg_d_lang['baseVsSilver'] = list(np.array(agg_d_lang['baseVsSilver']).mean(axis=0))
    agg_d_lang['ftVsSilver'] = list(np.array(agg_d_lang['ftVsSilver']).mean(axis=0))
    agg_d_lang['baseVsFt'] = list(np.array(agg_d_lang['baseVsFt']).mean(axis=0))

    agg_d_lang['baseVsSilver'].append(sum(agg_d_lang['baseVsSilver'])/len(agg_d_lang['baseVsSilver']))
    agg_d_lang['ftVsSilver'].append(sum(agg_d_lang['ftVsSilver'])/len(agg_d_lang['ftVsSilver']))
    agg_d_lang['baseVsFt'].append(sum(agg_d_lang['baseVsFt'])/len(agg_d_lang['baseVsFt']))
    agg_langs[lang] = agg_d_lang

with open("data/jsonData2/out/"+args.id+"/agg_sts.txt", 'w') as file:
    file.write(json.dumps(agg_d))

with open("data/jsonData2/out/"+args.id+"/agg_sts_langs.txt", 'w') as file:
    file.write(json.dumps(agg_langs))

if args.id in ['gn3','gn22']:
    output = "data/jsonData2/out/"+args.id+"/test_sts_chatgpt/*.txt"
    ml = glob.glob(output)
    agg_d = {"chatgptVsSilver": [], "chatgptVsFt": []}
    agg_langs = {'hindi': {"chatgptVsSilver": [], "chatgptVsFt": []}, 'tamil': {"chatgptVsSilver": [], "chatgptVsFt": []}, 'telugu': {"chatgptVsSilver": [], "chatgptVsFt": []}, 'urdu': {"chatgptVsSilver": [], "chatgptVsFt": []}}
    for fname in tqdm(ml, ncols=100, desc='chatgpt collecting'):
        try:
            with open(fname) as f:
                data = json.loads(f.read())
        except Exception as e:
            print(e)
            print(fname)
            print('-'*50,'error in reading file','-'*50)
            continue
        agg_d['chatgptVsSilver'].append(data['baseVsSilver'])
        agg_d['chatgptVsFt'].append(data['baseVsFt'])

        url = test_data[int(fname.split('/')[-1].replace('.txt',''))]['url']
        lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', url)[0].split('/')[1]
        agg_langs[lang]['chatgptVsSilver'].append(data['baseVsSilver'])
        agg_langs[lang]['chatgptVsFt'].append(data['baseVsFt'])
    
    assert len(agg_d['chatgptVsSilver']) == len(agg_d['chatgptVsFt'])
    assert len(agg_d['chatgptVsSilver']) == sum([len(agg_langs[x]['chatgptVsSilver']) for x in agg_langs.keys()])

    agg_d['chatgptVsSilver'] = list(np.array(agg_d['chatgptVsSilver']).mean(axis=0))
    agg_d['chatgptVsFt'] = list(np.array(agg_d['chatgptVsFt']).mean(axis=0))

    agg_d['chatgptVsSilver'].append(sum(agg_d['chatgptVsSilver'])/len(agg_d['chatgptVsSilver']))
    agg_d['chatgptVsFt'].append(sum(agg_d['chatgptVsFt'])/len(agg_d['chatgptVsFt']))

    print(agg_d)

    for lang in agg_langs.keys():
        agg_d_lang = agg_langs[lang]
        agg_d_lang['chatgptVsSilver'] = list(np.array(agg_d_lang['chatgptVsSilver']).mean(axis=0))
        agg_d_lang['chatgptVsFt'] = list(np.array(agg_d_lang['chatgptVsFt']).mean(axis=0))

        agg_d_lang['chatgptVsSilver'].append(sum(agg_d_lang['chatgptVsSilver'])/len(agg_d_lang['chatgptVsSilver']))
        agg_d_lang['chatgptVsFt'].append(sum(agg_d_lang['chatgptVsFt'])/len(agg_d_lang['chatgptVsFt']))
        agg_langs[lang] = agg_d_lang

    with open("data/jsonData2/out/"+args.id+"/agg_sts_chatgpt.txt", 'w') as file:
        file.write(json.dumps(agg_d))

    with open("data/jsonData2/out/"+args.id+"/agg_sts_chatgpt_langs.txt", 'w') as file:
        file.write(json.dumps(agg_langs))