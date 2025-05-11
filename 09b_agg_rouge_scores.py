# this code calculates the aggregate rouge scores given an experiment id
# t32v2

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

print('Agregating t32_v2.py',datetime.datetime.now(IST).strftime("%c"))

run = wandb.init(project="genmqna2_rouge_agg", id=args.id, name=args.id, resume='allow')

output = "data/jsonData2/out/"+args.id+"/test_rouge/*.txt"

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

test_file_path = id_to_test_file[args.id]
# read jsonl file
test_data = []
with open(test_file_path, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines, ncols=100, desc='Reading Test Data'):
        test_data.append(json.loads(line))

agg_d = {"baseVsSilver": {}, "ftVsSilver":{}, "baseVsFt": {}}
agg_langs = {'hindi': {"baseVsSilver": {}, "ftVsSilver":{}, "baseVsFt": {}}, 'tamil': {"baseVsSilver": {}, "ftVsSilver":{}, "baseVsFt": {}}, 'telugu': {"baseVsSilver": {}, "ftVsSilver":{}, "baseVsFt": {}}, 'urdu': {"baseVsSilver": {}, "ftVsSilver":{}, "baseVsFt": {}}}

for fname in tqdm(ml, ncols=100, desc='collecting'):
    try:
        with open(fname) as f:
            data = json.loads(f.read()) # {"baseVsSilver": {"rouge1": [f , f , f], "rouge2": [f , f , f], "rouge3": [f , f , f], "rougeL": [f , f , f]}, "ftVsSilver": {"rouge1": [f , f , f], "rouge2": [f , f , f], "rouge3": [f , f , f], "rougeL": [f , f , f]}, "baseVsFt": {"rouge1": [f , f , f], "rouge2": [f , f , f], "rouge3": [f , f , f], "rougeL": [f , f , f]}}
    except Exception as e:
        print(e)
        print(fname)
        print('-'*50,'error in reading file','-'*50)
        continue
    try:
        assert len(data.keys()) == 3
    except:
        print('\n\n',data)
        exit()
    for k in data.keys():
        assert k in agg_d.keys()
        for k2 in data[k].keys():
            if k2 not in agg_d[k].keys():
                agg_d[k][k2] = []
            agg_d[k][k2].append(data[k][k2])

    url = test_data[int(fname.split('/')[-1].split('.')[0])]['url']
    lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', url)[0].split('/')[1]
    # print('\n before',agg_langs)
    for k in data.keys():
        assert k in agg_langs[lang].keys()
        for k2 in data[k].keys():
            if k2 not in agg_langs[lang][k].keys():
                agg_langs[lang][k][k2] = []
            agg_langs[lang][k][k2].append(data[k][k2])
        # print('\n',data)
        # print('after',agg_langs)
        # input('wait')
try:
    assert len(agg_d['baseVsSilver']['rouge1']) == len(agg_d['ftVsSilver']['rouge1']) == len(agg_d['baseVsFt']['rouge1'])
except:
    print('\n\n',len(agg_d['baseVsSilver']['rouge1']), len(agg_d['ftVsSilver']['rouge1']), len(agg_d['baseVsFt']['rouge1']))
    exit()
assert len(agg_d['baseVsSilver']['rouge1']) == sum([len(agg_langs[x]['baseVsSilver']['rouge1']) for x in agg_langs.keys()])

# take average
for k in agg_d.keys():
    for k2 in agg_d[k].keys():
        # print(k2, str(agg_d[k][k2])[:100], np.array(agg_d[k][k2]).shape)
        agg_d[k][k2] = np.array(agg_d[k][k2]).mean(axis=0).tolist()

print(agg_d)

# take average lang wise
for lang in agg_langs.keys():
    agg_d_lang = agg_langs[lang]
    for k in agg_d_lang.keys():
        for k2 in agg_d_lang[k].keys():
            # print(k2, str(agg_d_lang[k][k2])[:100], np.array(agg_d_lang[k][k2]).shape)
            agg_d_lang[k][k2] = np.array(agg_d_lang[k][k2]).mean(axis=0).tolist()
    agg_langs[lang] = agg_d_lang

with open("data/jsonData2/out/"+args.id+"/agg_rouge.txt", 'w') as file:
    file.write(json.dumps(agg_d))

with open("data/jsonData2/out/"+args.id+"/agg_rouge_langs.txt", 'w') as file:
    file.write(json.dumps(agg_langs))

if args.id in ['gn3','gn22']:
    output = "data/jsonData2/out/"+args.id+"/test_rouge_chatgpt/*.txt"
    ml = glob.glob(output)
    agg_d = {"chatgptVsSilver": {}, "chatgptVsFt": {}}
    agg_langs = {'hindi': {"chatgptVsSilver": {}, "chatgptVsFt": {}}, 'tamil': {"chatgptVsSilver": {}, "chatgptVsFt": {}}, 'telugu': {"chatgptVsSilver": {}, "chatgptVsFt": {}}, 'urdu': {"chatgptVsSilver": {}, "chatgptVsFt": {}}}
    for fname in tqdm(ml, ncols=100, desc='chatgpt collecting'):
        try:
            with open(fname) as f:
                data = json.loads(f.read())
        except Exception as e:
            print(e)
            print(fname)
            print('-'*50,'error in reading file','-'*50)
            continue
        for k in data.keys():
            assert k in agg_d.keys()
            for k2 in data[k].keys():
                if k2 not in agg_d[k].keys():
                    agg_d[k][k2] = []
                agg_d[k][k2].append(data[k][k2])
        
        url = test_data[int(fname.split('/')[-1].split('.')[0])]['url']
        lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', url)[0].split('/')[1]
        for k in data.keys():
            assert k in agg_langs[lang].keys()
            for k2 in data[k].keys():
                if k2 not in agg_langs[lang][k].keys():
                    agg_langs[lang][k][k2] = []
                agg_langs[lang][k][k2].append(data[k][k2])
    assert len(agg_d['chatgptVsSilver']['rouge1']) == len(agg_d['chatgptVsFt']['rouge1'])
    assert len(agg_d['chatgptVsSilver']['rouge1']) == sum([len(agg_langs[x]['chatgptVsSilver']['rouge1']) for x in agg_langs.keys()])
    # take average
    for k in agg_d.keys():
        for k2 in agg_d[k].keys():
            agg_d[k][k2] = np.array(agg_d[k][k2]).mean(axis=0).tolist()
    print('chatgpt')
    print(agg_d)
    # take average lang wise
    for lang in agg_langs.keys():
        agg_d_lang = agg_langs[lang]
        for k in agg_d_lang.keys():
            for k2 in agg_d_lang[k].keys():
                agg_d_lang[k][k2] = np.array(agg_d_lang[k][k2]).mean(axis=0).tolist()
        agg_langs[lang] = agg_d_lang
    with open("data/jsonData2/out/"+args.id+"/agg_rouge_chatgpt.txt", 'w') as file:
        file.write(json.dumps(agg_d))
    with open("data/jsonData2/out/"+args.id+"/agg_rouge_chatgpt_langs.txt", 'w') as file:
        file.write(json.dumps(agg_langs))

    