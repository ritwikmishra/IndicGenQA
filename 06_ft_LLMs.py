# t21
import os 
import pickle
import json
import json
import os, glob
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
from tqdm import tqdm
import wandb
wandb.login(key=input('enter wandb key:'))

# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir='data/cache_models/', device_map="auto", trust_remote_code=True)
# peft_model_id = "/raid/home/ritwikm/git/mqna/data/jsonData/mannOut/checkpoint-60/"
# ## model = base_model
# model = PeftModel.from_pretrained(base_model, peft_model_id)
# model.merge_and_unload()
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# from transformers import GenerationConfig
# gc = GenerationConfig(
#             do_sample=False,
#             num_beams=2,
#             max_new_tokens=50,
#         )
# prompt = '''Answer the question based on the given context.

# ##Question
# What is the birth date of Sane Guruji?

# ##Context
# Sane was born on 24 December 1899 to Sadashivrao and Yashodabai Sane in Palgad village near Dapoli town, Bombay State in British India (in present-day Ratnagiri district of the Konkan region of Maharashtra state).

# ##Answer'''
# inputs = tokenizer(prompt, return_tensors="pt")
# inputs = {k: v.to("cuda") for k, v in inputs.items()}
# with torch.no_grad():
#     generate_ids = model.generate(**inputs, generation_config=gc)
# outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print('\n\n\n',outputs)

# exit()

# arguments
# python t21.py --model 'meta-llama/Llama-2-7b-chat-hf' --train 'data/jsonData/mann_train.jsonl' --val 'data/jsonData/mann_val.jsonl' --output data/jsonData/mannOut/g14/
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--train", type=str, help="Show Dataset File")
parser.add_argument("--val", type=str, help="Show Dataset File")
parser.add_argument("--output", type=str, help="Show Fined Tuned Folder", default='')
parser.add_argument("--max_seq_length", type=int, help="Show Fined Tuned Folder", default=9500) # 9500 gemma, 16000 llama-3.1
parser.add_argument("--resume_from_checkpoint", type=str, help="Show Fined Tuned Folder", default=True)
parser.add_argument("--check_token_dist",type=str, default='No')
parser.add_argument("--msg", type=str, default='')
args = parser.parse_args()
args.output = "data/jsonData2/out/"+args.id+"/"

ml = glob.glob(args.output+'*')
args.resume_from_checkpoint = False
for a in ml:
    if 'checkpoint' in a:
        args.resume_from_checkpoint = True
        break

run = wandb.init(project="genmqna2", id=args.id, name=args.id, resume='allow')



# for arg_name in vars(args):
#     arg_value = getattr(args, arg_name)
#     print(f'{arg_name}: {arg_value}')


MODEL_NAME = args.model
TRAIN = args.train 
VAL = args.val
OUTPUT_DIR = args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dataset = load_dataset("json", data_files=TRAIN)["train"]
valid_dataset = load_dataset("json", data_files=VAL)["train"] # unknown split error https://github.com/clovaai/donut/issues/57#issuecomment-2145467942

if 'CohereForAI' not in MODEL_NAME:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(len(train_dataset), len(valid_dataset))

if args.check_token_dist != 'No':
    prompt_len = []
    for item in tqdm(train_dataset, ncols=100, desc='train'):
        prompt_len.append(len(tokenizer.tokenize(item['prompt'])))

    # print percentiles of prompt lengths
    pvals = [12, 25, 50, 75, 90, 95, 99, 100]
    p = np.percentile(prompt_len, pvals)
    for p,v in zip(pvals, p):
        print(f'Train {p:3}: {v:3}')
        # wandb.log({f'train_prompt_len_{p}': v})
    exit()
# prompt_len = []
# for item in tqdm(valid_dataset, ncols=100, desc='valid'):
#     prompt_len.append(len(tokenizer.tokenize(item['prompt'])))

# # print percentiles of prompt lengths
# p = np.percentile(prompt_len, pvals)
# for p,v in zip(pvals, p):
#     print(f'Valid {p:3}: {v:3}')
#     # wandb.log({f'valid_prompt_len_{p}': v})


compute_dtype = getattr(torch, "float16")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # load_in_8bit=True,
    # quantization_config=quant_config,
    device_map="auto",
    # trust_remote_code=True, # removed for cohere
    cache_dir='data/cache_models/',
    # max_memory={0: "10GiB", 1: "10GiB", 2:"23GiB", 3: "23GiB", 4:"46GiB", 5:"10GiB", 6: "10GiB", "cpu": "230GiB"},
    # attn_implementation="eager"
    
)
# print(model)
model.config.use_cache = False
model.config.pretraining_tp = 1

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


peft_params = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj",       
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_params)
model.print_trainable_parameters()

EPOCHS = 1

BATCH_SIZE = 1
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300

training_params = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    # gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    per_device_eval_batch_size=MICRO_BATCH_SIZE,
    # warmup_steps=0.2,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=10,
    optim="adamw_hf",
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=0.2,
    save_steps=0.05,
    output_dir=OUTPUT_DIR,
    save_total_limit=EPOCHS,
    # load_best_model_at_end=True,
    logging_dir=OUTPUT_DIR,
    report_to="wandb"
)

def formatting_prompts_func(example):
    print(example)
    # output_texts = []
    # for i in range(len(example['question'])):
    #     text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
    #     output_texts.append(text)
    # return output_texts

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_params,
    # dataset_text_field="messages",
    formatting_func=formatting_prompts_func,
    max_seq_length=args.max_seq_length,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    dataset_text_field="prompt",
)

trainer.train(resume_from_checkpoint = args.resume_from_checkpoint) #resume_from_checkpoint = True