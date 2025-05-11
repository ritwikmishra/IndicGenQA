# this code run the inference on the test set of genmqna2 by CHATGPT models in base form t30v2
# since it is costly, we do it for a subset of the test set


# this is for the q-paragraph setting
from openai import OpenAI
import pandas as pd
import pickle
import json, os
from tqdm import tqdm
import time, json, re

import wandb
wandb.login(key=input('enter wandb key:'))

import tiktoken

import argparse, glob

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--json_file", type=str, help="json file to pick")
parser.add_argument("--check_tokens", type=str, help="calculates token cost of the prompt", default='no')
parser.add_argument("--till", type=int, help="prompt to test", default=-1)
args = parser.parse_args()

if args.check_tokens != 'yes':
    # Define file paths
    OPENAI_API_KEY = input('Enter OpenAI API Key: ')

    def generate(prompt, model_name, max_tokens, api_key=OPENAI_API_KEY):
        client = OpenAI(api_key=api_key)
        
        response =  client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.001,
            max_tokens=max_tokens,
            n=1,
            stop=None
        )

        return response.choices[0].message.content

run = wandb.init(project="genmqna2_chatgpt", id=args.id, name=args.id, resume='allow')

enc = tiktoken.encoding_for_model(args.model)
# read jsonl file
test_file_path = args.json_file
# read jsonl file
test_data = []
with open(test_file_path, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines, ncols=100, desc='Reading Test Data'):
        test_data.append(json.loads(line))

if args.till != -1:
    test_data = test_data[:args.till]

args_dict = vars(args)
print(json.dumps(args_dict, indent=4))
print(len(test_data))

if not os.path.exists('data/jsonData2/out/'+args.id+'/test_output_'+args.model):
    os.makedirs('data/jsonData2/out/'+args.id+'/test_output_'+args.model)

total_tokens = {'input':0, 'output':0}
lang_wise_tokens = {}
pbar = tqdm(enumerate(test_data), ncols=120, desc='running', total=len(test_data))
for i,item in pbar:    
    prompt = item['prompt'].split('##Answer')[0]+'##Answer\n'
    silver_answer = item['prompt'].split('##Answer')[1].strip()
    if args.check_tokens == 'yes':
        input_tokens = len(enc.encode(prompt))
        output_tokens = len(enc.encode(silver_answer))
        total_tokens['input']+=input_tokens
        total_tokens['output']+=output_tokens
        lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', item['url'])[0].split('/')[1]
        if lang not in lang_wise_tokens:
            lang_wise_tokens[lang] = {'input':0, 'output':0}
        lang_wise_tokens[lang]['input']+=input_tokens
        lang_wise_tokens[lang]['output']+=output_tokens
    else:
        if os.path.exists('data/jsonData2/out/'+args.id+'/test_output_'+args.model+'/'+str(i)+'.json'):
            # check if the file is readable
            with open('data/jsonData2/out/'+args.id+'/test_output_'+args.model+'/'+str(i)+'.json', 'r') as file:
                try:
                    file_content = json.loads(file.read())
                    assert 'response' in file_content
                    continue
                except:
                    pass
        t = len(enc.encode(silver_answer))
        response = generate(prompt, args.model, t)
        item['response'] = response
        #write item to a json file
        with open('data/jsonData2/out/'+args.id+'/test_output_'+args.model+'/'+str(i)+'.json', 'w') as file:
            json.dump(item, file, ensure_ascii=False)
        # input('wait')

if args.check_tokens == 'yes':
    # https://openai.com/api/pricing/
    input_cost = 0.15 # per million
    output_cost = 0.6 # per million
    print('Total Tokens:', total_tokens)
    # print total cost, each input and output and their sum. Round upto 4 decimal places.
    print('Total Cost for input tokens:', round(total_tokens['input']*input_cost/1e6, 4))
    print('Total Cost for output tokens:', round(total_tokens['output']*output_cost/1e6, 4))
    print('Total Cost:', round((total_tokens['input']*input_cost + total_tokens['output']*output_cost)/1e6, 4))
    print('Lang wise Tokens:', lang_wise_tokens)
    # print total cost, each input and output and their sum. Round upto 4 decimal places.
    print('Lang wise Cost for input tokens:', {k:round(v['input']*input_cost/1e6, 4) for k,v in lang_wise_tokens.items()})
    print('Lang wise Cost for output tokens:', {k:round(v['output']*output_cost/1e6, 4) for k,v in lang_wise_tokens.items()})
    print('Lang wise Total Cost:', {k:round((v['input']*input_cost + v['output']*output_cost)/1e6, 4) for k,v in lang_wise_tokens.items()})

    if 'train' in test_file_path:
        training_cost = 25.0 # per million
        print('Training Cost:', round(((total_tokens['input']+total_tokens['output'])*training_cost)/1e6, 4))




