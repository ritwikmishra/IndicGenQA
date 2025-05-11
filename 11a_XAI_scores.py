# this code calculates the explainability scores using our finetuned APS model
# t44

# model libs
import os, torch, numpy as np, random
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

# data libs
import pickle, glob, json, re
from tqdm import tqdm

hyper_params = {
    "run_ID": "17c3_2", # "17c3" -->xlmr,  "17g3"-->xlmv
    "message": " |d all big with class weights",
    "dummy_run": False,
    "rseed": 123,
    "bert_model_name": "xlm-roberta-base", # "facebook/xlm-v-base" "xlm-roberta-base"
    "nw": 0,
    "max_len": 512,
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "epochs": 1,
    "encoder_alpha": 1e-05,
    "task_alpha": 0.0003,
    "snapshot_path": "data/",
    "threshold": 0.5,
    "include_prev_context": True,
    "include_title": False,
    "datasize": "big",
    "loss": "wfl",
    "pos_embeddings": False,
    "posE_V": 4181,
    "posE_D": 20,
    "comparison": "para-level",
    "fl_gamma": 2.0,
    "save_every_x_iter": 5000,
    "batch_size_list": "[12, 12, 12, 12, 12]",
    "world_size": 5
}

signature = {
    'max_len':hyper_params['max_len'],
    'bert_model_name':hyper_params['bert_model_name'],
    'include_prev_context':hyper_params['include_prev_context'],
    'include_title':hyper_params['include_title'],
    'datasize':hyper_params['datasize'],
    'pos_embeddings':hyper_params['pos_embeddings'],
    'comparison':hyper_params['comparison'],
    'message':'all big with class weights'
}

os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
# Torch RNG
torch.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed_all(hyper_params['rseed'])
# Python RNG
np.random.seed(hyper_params['rseed'])
random.seed(hyper_params['rseed'])

class model_class(torch.nn.Module):
    def __init__(self,hyper_params, class_weights):
        super(model_class, self).__init__()
        self.hyper_params = hyper_params
        self.trainable = {}
        if 'google/mt5' in self.hyper_params['bert_model_name']:
            self.encoder = MT5EncoderModel.from_pretrained(self.hyper_params['bert_model_name'], cache_dir='data/cache_models', return_dict=True, output_hidden_states=True)
        else:
            self.encoder = AutoModel.from_pretrained(self.hyper_params['bert_model_name'], cache_dir='data/cache_models', return_dict=True, output_hidden_states=True)
        self.enc_config = AutoConfig.from_pretrained(self.hyper_params['bert_model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.hyper_params['bert_model_name'])
        self.trainable['encoder'] = self.encoder
        self.class_weights = class_weights
        if hyper_params['pos_embeddings']:
            self.pos_embeddings = nn.Embedding(hyper_params['posE_V']+2,hyper_params['posE_D'])
            self.trainable['pos_embeddings'] = self.pos_embeddings
            poslen = 20
        else:
            poslen = 0
        self.fine_tuning_layers = nn.Sequential(nn.Linear(self.enc_config.hidden_size+poslen,100),nn.ReLU(),nn.Dropout(0.2),nn.Linear(100,50),nn.ReLU(),nn.Dropout(0.2),nn.Linear(50,1),nn.Sigmoid())
        self.trainable['fine_tuning_layers'] = self.fine_tuning_layers
        self.gamma = hyper_params['fl_gamma']
        if hyper_params['loss'] == 'bce':
            self.loss_fun = nn.BCELoss()
        elif hyper_params['loss'] == 'wbce':
            self.loss_fun = self.BCELoss_ClassWeights
        elif hyper_params['loss'] == 'fl':
            self.loss_fun = self.FocLoss
        elif hyper_params['loss'] == 'wfl':
            self.loss_fun = self.FocLoss_ClassWeights
        else:
            raise 'error'
        self.best_val_metric = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
    
    def BCELoss_ClassWeights(self, logits, labels):
        class_weights = self.class_weights
        logits = torch.clamp(logits,min=1e-7,max=1-1e-7)
        bce = - ( class_weights[1] * labels * torch.log(logits) + class_weights[0] * (1 - labels) * torch.log(1 - logits) )
        return torch.mean(bce)
    
    def FocLoss_ClassWeights(self, logits, labels):
        class_weights = self.class_weights
        logits = torch.clamp(logits,min=1e-7,max=1-1e-7)
        bce = - ( class_weights[1] * labels * ((1-logits)**self.gamma) * torch.log(logits) + class_weights[0] * (1 - labels) * (logits**self.gamma) * torch.log(1 - logits) )
        return torch.mean(bce)
    
    def FocLoss(self, logits, labels):
        logits = torch.clamp(logits,min=1e-7,max=1-1e-7)
        bce = - ( labels * ((1-logits)**self.gamma) * torch.log(logits) + (1 - labels) * (logits**self.gamma) * torch.log(1 - logits) )
        return torch.mean(bce)

    def forward(self, input_ids, attention_mask, pos=None, labels=None, **kwargs):
        # clip input_ids, attention_mask to max_len
        input_ids = input_ids[:,:self.hyper_params['max_len']].to(self.device)
        attention_mask = attention_mask[:,:self.hyper_params['max_len']].to(self.device)

        # # temporarily conncat input ids with input ids again
        # input_ids = torch.cat([input_ids, input_ids], dim=0)
        # attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        # print(input_ids.shape)
        # input('wait')
        bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0] # embedding of cls token in every batch
        if hyper_params['pos_embeddings']:
            pos_e = self.pos_embeddings(pos)
            out = torch.concat([bert_output, pos_e],dim=1)
        else:
            out = bert_output
        logits = self.fine_tuning_layers(out)
        # print('one',logits.shape)
        # logits = logits.view(-1)
        # print('two',logits.shape)
        logits = torch.log(torch.cat([logits, 1-logits], axis=1))
        if labels is not None:
            assert logits.shape == labels.shape
            # print(logits.type(), labels.type())
            # input('wait')
            loss = self.loss_fun(logits, labels)
        else:
            loss = None
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    def predict(self, context, question):
        context_lines = context.split('\n')
        text, pos = [], []
        for i,_ in enumerate(context_lines):
            a_part = ' '.join(context_lines[:i+1])
            len_q_part = len(self.tokenizer.tokenize(question))
            a_part = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(a_part)[-(self.hyper_params['max_len']-len_q_part):])
            text.append(question+' '+a_part)
            pos.append(i)
        
        # tok = self.tokenizer(text, padding=True, truncation=True, max_length=self.hyper_params['max_len'], return_tensors='pt')
        # input_ids = tok['input_ids'].to(self.device)
        # attention_mask = tok['attention_mask'].to(self.device)
        # pos = torch.tensor(pos).to(self.device)
        # with torch.no_grad():
        #     logits = self.forward(input_ids, attention_mask, pos).logits
        logits = self.predict_this(text)
        # result = []
        # for line, score in zip(context_lines, logits.tolist()):
        #     result.append((line, score))
        return logits.tolist()

    def predict_this(self, text):
        logits_list = []
        offset = 8 # # reduce for small GPUs
        i = 0
        while i < len(text):
            tok = self.tokenizer(text[i:i+offset], padding=True, truncation=True, max_length=self.hyper_params['max_len'], return_tensors='pt')
            input_ids = tok['input_ids'].to(self.device)
            attention_mask = tok['attention_mask'].to(self.device)
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask, None).logits
            logits_list.append(logits.detach().cpu())
            i+=offset
        logits = torch.concat(logits_list, dim=0)
        return logits

aps_model = model_class(hyper_params, [])
aps_model = aps_model.to(aps_model.device)
aps_model.eval()

tokenizer = AutoTokenizer.from_pretrained(hyper_params['bert_model_name'])

# a = model.forward(**tokenizer('Who was the first prime minister of India? First prime minister of India was Jawaharlal Nehru.', return_tensors='pt', truncation=True, max_length=hyper_params['max_len']))

print('loading model',hyper_params['run_ID'])
snapshot = torch.load(hyper_params['snapshot_path']+'snapshot'+hyper_params['run_ID']+'.pt', map_location="cuda:0")
print(aps_model.load_state_dict(snapshot["MODEL_STATE"]))


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# name = "Hate-speech-CNERG/hindi-abusive-MuRIL"
# model = AutoModelForSequenceClassification.from_pretrained(name)
# tokenizer = AutoTokenizer.from_pretrained(name)

from ferret import Benchmark
from ferret.explainers.lime import LIMEExplainer
from ferret.explainers.shap import SHAPExplainer
import glob, re, pickle, os, json
import pandas as pd
from collections import Counter

bench = Benchmark(aps_model, tokenizer, explainers=[SHAPExplainer(aps_model, tokenizer),LIMEExplainer(aps_model, tokenizer)])

# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
with open('data/lang_to_code.json','r') as f:
    lang_to_code = json.load(f)

checkpoint = "facebook/nllb-200-1.3B" # "facebook/nllb-200-distilled-600M" 'facebook/nllb-200-distilled-1.3B' 'facebook/nllb-200-1.3B' 'facebook/nllb-200-3.3B'
#                                                                                                                                           OOM ^^

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
tokenizer2 = AutoTokenizer.from_pretrained(checkpoint)

print('Loading model')
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.eval()

pten_pipeline = pipeline('translation', model=model, tokenizer=tokenizer2, device="cuda:1")


print('Loading dataset')
with open('data/dataset/MuNfQuAD_v2.pkl','rb') as f:
    data = pickle.load(f)
print('Length of data:', len(data))

from collections import defaultdict
test_urls_dict = defaultdict(bool)

with open('data/test_urls_gen.pkl', 'rb') as f:
    test_urls = pickle.load(f)

for tu in tqdm(test_urls, ncols=100, desc='test_urls'):
    test_urls_dict[tu] = True

hindi_count = len(glob.glob('data/ferret_xai/*.json'))

pbar = tqdm(enumerate(data), ncols=100, total=len(data), desc='data')

for arti,art in pbar:
    if test_urls_dict[art['url']]:
        if os.path.exists(f'data/ferret_xai/'+str(arti)+'.json'):
            with open(f'data/ferret_xai/'+str(arti)+'.json','r') as f:
                json_list = json.load(f)
            prediction_added = False
            for k in json_list.keys():
                for qas in json_list[k]:
                    for text_dict in qas:
                        if 'prediction' not in text_dict:
                            pbar.set_description(f're-Predicting {arti}')
                            text = ' '.join(text_dict['Token'][1:-1])
                            pred = aps_model.forward(**tokenizer(text, return_tensors='pt', truncation=True, max_length=hyper_params['max_len']))
                            text_dict['prediction'] = pred.logits.tolist()
                            prediction_added = True
            if prediction_added:
                assert 'prediction' in json_list[list(json_list.keys())[0]][0][0]
                assert 'prediction' in json_list['english'][0][0]
                with open(f'data/ferret_xai/'+str(arti)+'.json','w') as f:
                    json.dump(json_list, f, ensure_ascii=False)
            continue
        pbar.set_description(f'data Found {arti}')
        language = re.search(r'bbc\..*?\/\w+',art['url'])[0].split('/')[1]
        if language != 'hindi' and hindi_count < 5:
            continue
        src_code = lang_to_code[language][0]
        context_paras = art['paragraphs'][0]['context'].split('\n')
        sections = [int(x) for x in art['paragraphs'][0]['sections'].split(',')]
        url = art['url']
        json_list = {language:[], 'english':[]}
        try:
            for qasi,qas in enumerate(art['paragraphs'][0]['qas']):
                qas_list = []
                qas_list_en = []
                question = qas['question']
                text_list = [ question + ' ' + c for c in context_paras ]
                trimmed_context = art['paragraphs'][0]['context'][:qas['answers'][0]['answer_start']].strip()
                a_section = sections[len(trimmed_context.split('\n'))]
                for texti,text in enumerate(text_list):
                    pbar.set_description(f"Processing {arti} qas {qasi+1}/{len(art['paragraphs'][0]['qas'])} c {texti+1}/{len(text_list)} len(text):{len(text.split())}")
                    explanations = bench.explain(text, target=1, show_progress=False)
                    df = bench.get_dataframe(explanations)
                    # print('\n\n',df)
                    a = tokenizer(text).word_ids()
                    # print(a, len(a))
                    a = a[1:-1]
                    a = [-1] + a
                    a = a + [len(a)]
                    # print(a, len(a), len(text.split()))
                    word_break = list(dict(sorted(Counter(a).items())).values())
                    # print(word_break)
                    # input('wait')
                    json_obj = {}
                    json_obj['Token'] = ['<begin>']+text.split()+['<end>']
                    for i,row in df.iterrows():
                        data = [row.name]
                        idx = 0
                        for offset in word_break:
                            ml = list(row[df.columns[idx:idx+offset]])
                            val = sum(ml)/len(ml) # averaging
                            data.append(val)
                            idx+=offset
                        json_obj[row.name] = data[1:]
                    json_obj['url'] = url
                    json_obj['qid'] = qas['id']
                    json_obj['label'] = 1.0 if sections[texti]==a_section else 0.0
                    pred = aps_model.forward(**tokenizer(text, return_tensors='pt', truncation=True, max_length=hyper_params['max_len']))
                    json_obj['prediction'] = pred.logits.tolist()
                    qas_list.append(json_obj)
                    pbar.set_description(f"Translating {arti} qas {qasi+1}/{len(art['paragraphs'][0]['qas'])} c {texti+1}/{len(text_list)} len(text):{len(text.split())}")
                    text_en = pten_pipeline(text, src_lang=src_code,tgt_lang='eng_Latn', max_length=1000)[0]['translation_text'].strip()
                    pbar.set_description(f"Processing_en {arti} qas {qasi+1}/{len(art['paragraphs'][0]['qas'])} c {texti+1}/{len(text_list)} len(text_en):{len(text_en.split())}")
                    explanations = bench.explain(text_en, target=1, show_progress=False)
                    df = bench.get_dataframe(explanations)
                    a = tokenizer(text_en).word_ids()
                    a = a[1:-1]
                    a = [-1] + a
                    a = a + [len(a)]
                    word_break = list(dict(sorted(Counter(a).items())).values())
                    json_obj_en = {}
                    json_obj_en['Token'] = ['<begin>']+text_en.split()+['<end>']
                    for i,row in df.iterrows():
                        data = [row.name]
                        idx = 0
                        for offset in word_break:
                            ml = list(row[df.columns[idx:idx+offset]])
                            val = sum(ml)/len(ml)
                            data.append(val)
                            idx+=offset
                        json_obj_en[row.name] = data[1:]
                    json_obj_en['url'] = url
                    json_obj_en['qid'] = qas['id']
                    json_obj_en['label'] = 1.0 if sections[texti]==a_section else 0.0
                    pred = aps_model.forward(**tokenizer(text_en, return_tensors='pt', truncation=True, max_length=hyper_params['max_len']))
                    json_obj_en['prediction'] = pred.logits.tolist()
                    qas_list_en.append(json_obj_en)

                json_list[language].append(qas_list)
                json_list['english'].append(qas_list_en)
        except Exception as e:
            print(e)
            continue
        with open(f'data/ferret_xai/'+str(arti)+'.json','w') as f:
            json.dump(json_list, f, ensure_ascii=False)
        
        if language == 'hindi':
            hindi_count+=1
    else:
        continue



