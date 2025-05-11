# this code run the inference on the test set of genmqna2 t30

import os 
import pickle
import json
import os, glob, time
# import bitsandbytes as bnb
import pandas as pd
import torch, numpy as np
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig
from tqdm import tqdm

def load_model(model_name):
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir='data/cache_models/', quantization_config=double_quant_config)
    if 'gemma-7b-it' in model_name:
        model = model.bfloat16() # https://github.com/facebookresearch/llama/issues/380#issuecomment-1656569519
    model.eval()
    return model

import wandb
wandb.login(key=input("Enter your wandb key: "))

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--load_chkpts", type=str, help="Show Fined Tuned Folder", default='yes')
parser.add_argument("--test", type=str, help="Show Dataset File")
parser.add_argument("--start_at", type=int, default=0)
parser.add_argument("--stop_at", type=int, default=7000)
parser.add_argument("--time", type=str, default='')
parser.add_argument('--this_split', type=str, default="1/1")
args = parser.parse_args()

this_split = int(args.this_split.split('/')[0])
total_splits = int(args.this_split.split('/')[1])

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

assert id_to_test_file[args.id] == args.test

output = "data/jsonData2/out/"+args.id+"/"

if 'm' in args.time:
    args.time = args.time.replace('m','')
    secs = int(args.time)*60
elif 'h' in args.time:
    args.time = args.time.replace('h','')
    secs = int(args.time)*3600
else:
    secs = 3*24*3600

ml = glob.glob(output+'*')
resume_from_checkpoint = ''
if args.load_chkpts == 'yes':
    for a in sorted(ml):
        if 'checkpoint' in a:
            resume_from_checkpoint = a
            # break

run = wandb.init(project="genmqna2_test", id=args.id, name=args.id, resume='allow')
a = time.time()
if resume_from_checkpoint:
    suffix = '_ft'
    print('#'*100,'\nWith checkpoint\n','#'*100)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir='data/cache_models/', device_map="auto").eval()
    peft_model_id = resume_from_checkpoint
    ## model = base_model
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.merge_and_unload()
    print('-'*50, 'checkpoint loaded', '-'*50)
else:
    suffix = '_base'
    print('#'*100,'\nWithout checkpoint\n','#'*100)
    model = load_model(args.model)
print('model loaded in ', time.time()-a, 'secs')
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
from transformers import GenerationConfig
gc = GenerationConfig(
            do_sample=False,
            num_beams=2,
            max_new_tokens=500,
        )

# que = 'What is the birth date of Sane Guruji?'
# context = 'Pandurang Sadashiv Sane, also known as Sane Guruji (Guruji meaning "respected teacher") by his students and followers, was a Marathi author, teacher, social activist and freedom fighter from Maharashtra, India. Sane\'s father Sadashivrao was a supporter of Lokmanya Tilak. However, after being imprisoned for a few days, he preferred to keep away from political matters. Sane resigned from his school job to join the Indian Independence Movement when Mahatma Gandhi started the Dandi March in 1930. He was imprisoned by the British authorities in the Dhule Jail for more than fifteen months for his work in the Civil Disobedience Movement. In 1932, Vinoba Bhave was in the same jail as Sane. Bhave delivered a series of lectures on the Bhagavad Gita each Sunday morning. Bhave\'s work Gītā Pravachane (Marathi: गीता प्रवचने) was an outcome of the notes Sane had made while imprisoned. However, Sane Guruji\'s mother proved to be a great influence in his life. He graduated with a degree in Marathi and Sanskrit and earned a master\'s degree in philosophy, before opting for a teaching profession. Sane worked as the teacher in Pratap High School in Amalner town. His literature was aimed at educating children. After Gandhi\'s assassination, he became very upset. He then died due to overdose of his sleeping pills. Sane was born on 24 December 1899 to Sadashivrao and Yashodabai Sane in Palgad village near Dapoli town, Bombay State in British India (in present-day Ratnagiri district of the Konkan region of Maharashtra state). Sane completed his primary education in the village of Dondaicha, in the Shindkheda taluka in Dhule district. After his primary education, he was sent to Pune to live with his maternal uncle for further education. However, he did not like his stay in Pune and returned to Palgadh to stay at a missionary school in Dapoli, about six miles from Palgad. While at Dapoli, he was quickly recognised as an intelligent student with good command over both the Marathi and Sanskrit languages. He was also interested in poetry. While in school at Dapoli, the financial condition of his family deteriorated further and he could not afford to continue his education. Like his elder brother, he considered taking up a job to help with the family finances. However, on the recommendation of one of his friends, and with support from his parents, he enrolled at the Aundh Institution, which provided free education and food to poor students. Here at Aundh he suffered many hardships but continued his education. However, an epidemic of bubonic plague in Aundh led to all students being sent home. '
# ans = ''
# prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{context}\n\n##Answer\n{ans}'
# inputs = tokenizer(prompt, return_tensors="pt")
# inputs = {k: v.to("cuda") for k, v in inputs.items()}
# with torch.no_grad():
#     generate_ids = model.generate(**inputs, generation_config=gc)
# outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print('\n\n\n',outputs)

# if directory doesn't exist, create it
if not os.path.exists(output+'test_output'+suffix):
    os.makedirs(output+'test_output'+suffix)

out_path = output+'test_output'+suffix+'/'

test_file_path = args.test
# read jsonl file
test_data = []
with open(test_file_path, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines, ncols=100, desc='Reading Test Data'):
        test_data.append(json.loads(line))

# load counter
# if os.path.exists(out_path+'count.pkl'):
#     if args.start_at == 0:
#         with open(out_path+'count.pkl', 'rb') as file:
#             args.start_at = pickle.load(file)
#     else:
#         print('start_at is not 0, so count.pkl is not loaded')
#         exit()
# if os.path.exists(out_path+'count_ckpt_'+args.load_chkpts+'.pkl'):
#     with open(out_path+'count_ckpt_'+args.load_chkpts+'.pkl', 'rb') as file:
#         args.start_at = pickle.load(file)
    # if args.start_at == 0:
    #     with open(out_path+'count_ckpt_'+args.load_chkpts+'.pkl', 'rb') as file:
    #         args.start_at = pickle.load(file)
    # else:
    #     print('start_at is not 0, so count.pkl is not loaded')
    #     exit()

print('start_at:', args.start_at)
print('stop_at:', args.stop_at)
print('time:', secs, 'secs')
current_time = time.time()
test_output_list = []
pbar = tqdm(enumerate(test_data), ncols=120, desc='running', total=len(test_data))
for i,item in pbar:
    if i % total_splits != this_split - 1:
        continue
    # exit if time exceeded
    if time.time()-current_time > secs:
        print('Time exceeded')
        breakthis_split
    # count how many files have been written using glob
    written = len(glob.glob(out_path+'*.txt'))
    if written >= args.stop_at:
        print('Stop at reached')
        break
    # save ith counter # later did not see the point of this
    # with open(out_path+'count_ckpt_'+args.load_chkpts+'.pkl', 'wb') as file:
    #     pickle.dump(i, file)
    # if i < args.start_at:
    #     pbar.set_description(f'running: Skipped')
    #     continue
    if os.path.exists(out_path+str(i)+'.txt'):
        with open(out_path+str(i)+'.txt', 'r') as file:
            outputs = file.read()
            test_output_list.append(outputs)
            pbar.set_description(f'running: Did')
            continue
    pbar.set_description(f'running / {written} written')
    prompt = item['prompt'].split('##Answer')[0]+'##Answer\n'
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    try:
        with torch.no_grad():
            generate_ids = model.generate(**inputs, generation_config=gc)
    except Exception as e:
        pbar.set_description(f'running: Error '+str(e))
        continue
    pbar.set_description(f'running: Done / {written} written')
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    test_output_list.append(outputs)
    with open(out_path+str(i)+'.txt', 'w') as file:
        file.write(outputs)

#save pickle
# with open(output+'combined_test_output'+suffix+'.pkl', 'wb') as file:
#     pickle.dump(test_output_list, file)