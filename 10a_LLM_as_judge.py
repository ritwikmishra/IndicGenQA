# this code treats chatgpt LLM as human judge
# t40
from openai import OpenAI
import pandas as pd
import pickle
import json, os
from tqdm import tqdm
import time, json, re, random
# fix a random seed
random.seed(123)

import wandb
wandb.login(key=input('enter wandb key:'))

import tiktoken
import argparse, glob

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--check_tokens", type=str, help="calculates token cost of the prompt", default='no')
parser.add_argument("--till", type=int, help="prompt to test", default=-1)
args = parser.parse_args()

run = wandb.init(project="genmqna2_chatgpt", id=args.id, name=args.id, resume='allow')

if args.check_tokens != 'yes':
    # Define file paths
    OPENAI_API_KEY = input('Enter OpenAI API Key: ')

    def generate(prompt, model_name, max_tokens, api_key=OPENAI_API_KEY):
        client = OpenAI(api_key=api_key)
        
        response =  client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.001,
            max_tokens=10,
            n=1,
            stop=None
        )

        return response.choices[0].message.content

args_dict = vars(args)
print(json.dumps(args_dict, indent=4))

enc = tiktoken.encoding_for_model(args.model)

filea = 'data/jsonData2/indie_gen_test_setting0.jsonl'
fileb = 'data/jsonData2/indie_gen_test_setting5.jsonl'
file1 = 'data/jsonData2/out/gn'+args.id.replace('gn','').split('v')[1]+'/test_output_ft/'
file2 = 'data/jsonData2/out/gn'+args.id.replace('gn','').split('v')[0]+'/test_output_ft/'

with open(filea, 'r') as file:
    lines = file.readlines()
    test_data1 = [json.loads(line) for line in lines]

with open(fileb, 'r') as file:
    lines = file.readlines()
    test_data2 = [json.loads(line) for line in lines]

bl = [x.split('/')[-1] for x in glob.glob(file1+'*.txt')]
ftl = [x.split('/')[-1] for x in glob.glob(file2+'*.txt')]

print('Test File1:', filea)
print('Test File2:', fileb)
print('File1:', file1)
print('File2:', file2)

print('Length of base list:', len(bl))
print('Length of fine-tuned list:', len(ftl))
common = sorted(list(set(bl).intersection(ftl)), key = lambda x: int(x.split('.')[0]))
print('Length of common list:', len(common))

if args.till != -1:
    common = common[:args.till]

if not os.path.exists('data/jsonData2/out/'+args.id+'/test_comparison_'+args.model):
    # makedirs recursively creates directories
    os.makedirs('data/jsonData2/out/'+args.id+'/test_comparison_'+args.model)

total_tokens = {'input':0, 'output':0}
lang_wise_tokens = {}
pbar = tqdm(enumerate(common), ncols=100, desc='Comparing', total=len(common))
for filei, c in pbar:
    cnum = int(c.split('.')[0])
    silver_answer1 = test_data1[cnum]['prompt'].split('##Answer')[1].split('##')[0].strip()
    silver_answer2 = test_data2[cnum]['prompt'].split('##Answer')[1].split('##')[0].strip()
    assert silver_answer1 == silver_answer2
    question = test_data1[cnum]['prompt'].split('##Question')[1].split('##')[0].strip()
    lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', test_data1[cnum]['url'])[0].split('/')[1]
    with open(file1+c, 'r') as file:
        opt1 = file.read()
    with open(file2+c, 'r') as file:
        opt2 = file.read()
    option1 = opt1.split('##Answer')[1].split('##')[0].strip()
    option2 = opt2.split('##Answer')[1].split('##')[0].strip()

    # sample a number between 0 and 1, if it is less than 0.5 then swap the options
    if random.random() < 0.5:
        option1, option2 = option2, option1
        suffix = 'option1 is '+file2+'\noption2 is '+file1
    else:
        suffix = 'option1 is '+file1+'\noption2 is '+file2

    # print('\nSilver Answer:', silver_answer)
    # print('\n\nOption1:', option1)
    # print('\n\nOption2:', option2)
    prompt = f'Given the following question, you are given a silver answer and two options. Choose the option that is closest to the silver answer. You are only allowed to choose one option. Print either "option1" or "option2". Print nothing else.\n\n##Question\n{question}\n\n##Silver_Answer: {silver_answer1}\n\n##Option1: {option1}\n\n##Option2: {option2}'
    if args.check_tokens == 'yes':
        input_tokens = len(enc.encode(prompt))
        output_tokens = 10
        total_tokens['input']+=input_tokens
        total_tokens['output']+=output_tokens
        if lang not in lang_wise_tokens:
            lang_wise_tokens[lang] = {'input':0, 'output':0}
        lang_wise_tokens[lang]['input']+=input_tokens
        lang_wise_tokens[lang]['output']+=output_tokens
    else:
        if os.path.exists('data/jsonData2/out/'+args.id+'/test_comparison_'+args.model+'/'+c.replace('.txt', '.json')):
            # check if the file is readable
            with open('data/jsonData2/out/'+args.id+'/test_comparison_'+args.model+'/'+c.replace('.txt', '.json'), 'r') as file:
                try:
                    file_content = json.loads(file.read())
                    assert 'response' in file_content
                    continue
                except:
                    pass
        item = {'prompt':prompt}
        item['url'] = test_data1[cnum]['url']
        item['qid'] = test_data1[cnum]['qid']
        response = generate(prompt, args.model, 10)
        # print(response)
        # print(suffix)        
        item['response'] = response+'\n'+suffix
        #write item to a json file
        with open('data/jsonData2/out/'+args.id+'/test_comparison_'+args.model+'/'+c.replace('.txt', '.json'), 'w') as file:
            json.dump(item, file, ensure_ascii=False)
        # input('wait')

if args.check_tokens == 'yes':
    # https://openai.com/api/pricing/
    input_cost = 2.5 # per million
    output_cost = 10.0 # per million
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