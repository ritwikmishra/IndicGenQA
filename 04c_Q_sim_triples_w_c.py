
# find similarity between question text and triples t25
# triples with coreference chains

# triples nomenclature is like this
# [a,b,c,d] means ath paragraph, bth sentence, cth triple, dth entity [0-2]
# this nomenclature will work for IndIE but for gen2oie b will always be zero

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

    def forward(self, input_ids, attention_mask, pos, labels=None):
        bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0] # embedding of cls token in every batch
        if hyper_params['pos_embeddings']:
            pos_e = self.pos_embeddings(pos)
            out = torch.concat([bert_output, pos_e],dim=1)
        else:
            out = bert_output
        logits = self.fine_tuning_layers(out).view(-1)
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

model = model_class(hyper_params, [])
model = model.to(model.device)
model.eval()

print('loading model',hyper_params['run_ID'])
snapshot = torch.load(hyper_params['snapshot_path']+'snapshot'+hyper_params['run_ID']+'.pt', map_location="cuda:0")
print(model.load_state_dict(snapshot["MODEL_STATE"]))


if not os.path.exists('data/qdict.pkl'):
    print('loading data')
    with open('data/data.pkl','rb') as f:
        data = pickle.load(f)
    print('data loaded')

    qdict = {}
    for art in tqdm(data, ncols=100):
        if art['url'] not in qdict:
            qdict[art['url']] = {}
        for qas in art['paragraphs'][0]['qas']:
            qdict[art['url']][qas['id']] = qas['question']
        
    with open('data/qdict.pkl','wb') as f:
        pickle.dump(qdict,f)
else:
    print('loading qdict')
    with open('data/qdict.pkl','rb') as f:
        qdict = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(hyper_params['bert_model_name'])

ml = glob.glob('data/gen2oie_data2/*.json')
# ttfname = 'data/text_tokens_gen2oie.pkl' # 95 percentile of token length (text = q + p) is less than 512

text_tokens = []

new_key = 'q_coref_chain_para_scores_2'

for fi,fname in tqdm(enumerate(ml), ncols=100, total=len(ml), desc='Making text_tokens'):
    # print('\n',fname)
    with open(fname) as f:
        oie_data = json.load(f)
    if 'res_data' not in oie_data:
        continue
    if new_key in oie_data:
        continue
    # print('\n',fname)
    # input('wait')
    url = oie_data['url']
    qids = oie_data['qids']
    coref_chain_paras = []
    for span_cluster in oie_data['res_data']:
        this_para = ''
        for span in span_cluster:
            if 'gen2oie' in fname:
                triple = oie_data['context_triples'][span[0]][span[2]]
            else:
                triple = oie_data['context_triples'][span[0]][span[1]][span[2]]
            this_para += (' '.join(triple)+'. ')
        coref_chain_paras.append(this_para)
    oie_data['coref_chain_paras'] = coref_chain_paras
    
    for qid in qids:
        text_list = []
        try:
            question = qdict[url][qid]
        except:
            continue 
        for para in coref_chain_paras:
            text = question + ' ' + para
            text_list.append(text)
            # text_tokens.append(len(tokenizer.tokenize(text)))
        result = model.predict_this(text_list).tolist()
        if new_key not in oie_data:
            oie_data[new_key] = {}
        oie_data[new_key][qid] = [question]+result
    
    with open(fname,'w',encoding='utf-8') as f:
        json.dump(oie_data,f,ensure_ascii=False,indent=4)
    # input('wait')

## find percentiles of text_tokens
# print('percentiles')
# for i in range(0,101,5):
#     print(i, np.percentile(text_tokens,i))
        
        
    