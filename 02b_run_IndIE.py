# this code runs IndIE on questions and contexts on entire MuNfQuAD t17

import argparse, json, pickle
from tqdm import tqdm
import time
import stanza

from chunking.chunking_model import *
from chunking.crf_chunker import *
from utils_40 import *

parser = argparse.ArgumentParser(description='Indie on the entire data')
parser.add_argument('--this_split', type=str)
args = parser.parse_args()

this_split = int(args.this_split.split('/')[0])
total_splits = int(args.this_split.split('/')[1])

lang = 'hi'
use_gpu = True

hyper_params = {
    'run_ID': 26,
    'createData': True,
    'bs': 8,
    'bert_model_name': 'xlm-roberta-base',
    'available_models': "'ai4bharat/indic-bert' 'bert-base-multilingual-cased' 'xlm-roberta-base' 'bert-base-multilingual-uncased' 'xlm-mlm-100-1280' 'xlm-mlm-tlm-xnli15-1024' 'xlm-mlm-xnli15-1024'",
    'alpha' : 0.00001,
    'epochs': 3,
    'rseed': 123,
    'nw': 4,
    'train_ratio' :  0.7,
    'val_ratio' : 0.1,
    'max_len' : 275,
    'which_way' : 3,
    'which_way_gloss': "1= take first wordpiece token id | 2= take last wordpiece token id | 3= average out the wordpiece embeddings during the forward pass",
    'embedding_way' : 'last_hidden_state',
    'embedding_way_gloss': 'last_hidden_state, first_two, last_two',
    'notation' : 'BI',
    'platform': 1,
    'available_platforms': "MIDAS server = 1, colab = 2",
    'chunker':'XLM' # CRF or XLM
}

model_embeddings = {
    'ai4bharat/indic-bert':768,
    'bert-base-multilingual-cased':768,
    'xlm-roberta-base':768,
    'bert-base-multilingual-uncased':768,
    'xlm-mlm-100-1280':1280,
    'xlm-mlm-tlm-xnli15-1024':1024,
    'xlm-mlm-xnli15-1024':1024
}

hyper_params['embedding_size'] = model_embeddings[hyper_params['bert_model_name']]

my_tagset = torch.load('chunking/data/my_tagset_'+hyper_params['notation']+'.bin')

hyper_params['my_tagset'] = my_tagset

os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
# Torch RNG
torch.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed_all(hyper_params['rseed'])
# Python RNG
np.random.seed(hyper_params['rseed'])
random.seed(hyper_params['rseed'])

is_cuda = torch.cuda.is_available()

if is_cuda and use_gpu:
    device = torch.device("cuda:0")
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t -(c-a)  # free inside cache
    print("GPU is available", torch.cuda.get_device_name(), round(t/(1024*1024*1024)), "GB")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

hyper_params['device'] = str(device)

hi_nlp = load_stanza_model('hi', use_gpu)
ur_nlp = load_stanza_model('ur', use_gpu)
tm_nlp = load_stanza_model('ta', use_gpu)
te_nlp = load_stanza_model('te', use_gpu)
en_nlp = load_stanza_model('en', use_gpu)
# exit()

if hyper_params['chunker'] == 'XLM':
    print('Creating the XLM chunker model...')
    model = chunker_class(device, hyper_params).to(device)
    # checkpoint = torch.load('chunking/state_dicts/model/'+str(hyper_params['run_ID'])+'_epoch_4.pth.tar',map_location=device)
    checkpoint = torch.load('chunking/state_dicts/model/26_repeat4_best.pth.tar',map_location=device)
    print(model.load_state_dict(checkpoint['state_dict'], strict=False))
    tokenizer = AutoTokenizer.from_pretrained(hyper_params['bert_model_name'])
elif hyper_params['chunker'] == 'CRF':
    model, tokenizer = 'CRF', 'CRF'

print('Loading dataset')
with open('data/dataset/MuNfQuAD_v2.pkl','rb') as f:
    data = pickle.load(f)

print('Train urls loading')
with open('data/train_urls.pkl','rb') as f:
    train_urls = pickle.load(f)

#convert it into default dictionary
train_url_dict = defaultdict(bool)

for tu in tqdm(train_urls, ncols=100, desc='prep'):
    short_url = re.search(r'www\.bbc\.(com|co\.uk)/.+', tu)[0].split('?')[0]
    train_url_dict[short_url] = True

data2 = []
for art in tqdm(data, ncols=100, desc='filtering'):
    if '/hindi/' not in art['url'] and '/tamil/' not in art['url'] and '/telugu/' not in art['url'] and '/urdu/' not in art['url'] and not re.search(r'bbc\.(com|co\.uk)\/news\/', art['url']):
        continue
    # if not re.search(r'bbc\.(com|co\.uk)\/news\/', art['url']):
    #     continue
    data2.append(art)
del data

errorneous_examples = {'hi':[], 'ta':[], 'te':[], 'ur':[], 'en':[]}

pbar = tqdm(enumerate(data2), ncols=100, total=len(data2), desc='Indie')
for arti,art in pbar:
    time.sleep(0.0000001)
    if re.search(r'bbc\.(com|co\.uk)\/news\/', art['url']):
        continue # skipping english for now
    if arti%total_splits != this_split-1:
        continue
    if train_url_dict[re.search(r'www\.bbc\.(com|co\.uk)/.+', art['url'])[0].split('?')[0]]:
        continue
    if os.path.exists(f'data/indie_data2/{arti}.json'):
        try:
            with open(f'data/indie_data2/{arti}.json', 'r') as f:
                res_art = json.load(f)
            assert res_art['url'] == art['url'] and res_art['context'] == art['paragraphs'][0]['context']
        except:
            pass
        else:
            continue
    ind_art = {'url':'', 'context':'', 'context_triples':[], 'qids':[], 'qtriples':[]}
    
    if '/hindi/' in art['url']:
        nlp = hi_nlp
        lang = 'hi'
    elif '/tamil/' in art['url']:
        nlp = tm_nlp
        lang = 'ta'
    elif '/telugu/' in art['url']:
        nlp = te_nlp
        lang = 'te'
    elif '/urdu/' in art['url']:
        nlp = ur_nlp
        lang = 'ur'
    else:
        nlp = en_nlp
        lang = 'en'
    
    context = art['paragraphs'][0]['context']
    context_paras = context.split('\n')
    context_triples = []
    try:
        for para in context_paras:
            _, context_para_triples, _, _ = perform_extraction(para,lang,model,tokenizer, nlp,show=False)
            context_triples.append(context_para_triples)
        # _, context_para_triples, _, _ = perform_extraction(context,lang,model,tokenizer, nlp,show=False)
        for qas in art['paragraphs'][0]['qas']:
            question = qas['question']
            _, exts, _, _ = perform_extraction(question,lang,model,tokenizer, nlp,show=False)
            ind_art['qids'].append(qas['id'])
            ind_art['qtriples'].append(exts)
            # print('\n\n')
            # print(context[:50])
            # print(str(exts)[:50])
            # input('wait')
    except Exception as e:
        # print(e)
        # if 'CUDA out of memory' in str(e):
        #     continue
        # else:
        #     raise e
        errorneous_examples[lang].append(art['url'])
    else:
        ind_art['url'] = art['url']
        ind_art['context'] = context
        ind_art['context_triples'] = context_triples

        with open(f'data/indie_data2/{arti}.json', 'w', encoding='utf-8') as f:
            json.dump(ind_art, f, ensure_ascii=False, indent=4)

print('Total number of errorneous examples:')
for k in errorneous_examples:
    print(k, len(errorneous_examples[k]))

with open('data/errorneous_examples.json', 'w') as f:
    json.dump(errorneous_examples, f, ensure_ascii=False, indent=4)

