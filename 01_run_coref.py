# performs coreference resolution on the entire data t16

from mcoref import *
import argparse, json, pickle
from tqdm import tqdm
import time
from collections import defaultdict

coref_model = CorefModel("coref/config.toml", "xlmr")
coref_model.load_weights(path="/media/data_dump/Ritwik/git/wl-coref/data2/xlmr_multi_plus_hi2.pt", map_location="cuda:0",ignore={"bert_optimizer", "general_optimizer","bert_scheduler", "general_scheduler"})
coref_model.training = False

parser = argparse.ArgumentParser(description='Coreference resolution on the entire data')
parser.add_argument('--this_split', type=str)
args = parser.parse_args()

this_split = int(args.this_split.split('/')[0])
total_splits = int(args.this_split.split('/')[1])

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

# data = [data[i] for i in tqdm(range(len(data)), desc='splitting') if i%total_splits == this_split-1]
pbar = tqdm(enumerate(data), ncols=100, total=len(data), desc='coref')
for arti,art in pbar:
    time.sleep(0.0000001)
    if train_url_dict[re.search(r'www\.bbc\.(com|co\.uk)/.+', art['url'])[0].split('?')[0]]:
        continue
    if arti%total_splits != this_split-1:
        continue
    if os.path.exists(f'data/res_data/{arti}.json'):
        try:
            with open(f'data/res_data/{arti}.json', 'r') as f:
                res_art = json.load(f)
            assert res_art['url'] == art['url'] and res_art['context'] == art['paragraphs'][0]['context']
        except:
            pass
        else:
            continue
    res_art = {'url':'', 'context':'', 'docs':''}
    context = art['paragraphs'][0]['context']
    pbar.set_description(f'coref arti={arti}, len(context)={len(context)}')
    try:
        docs = resolve_coreferences(context.split('\n'), coref_model)
    except Exception as e:
        if 'CUDA out of memory' in str(e):
            continue
        else:
            raise e
    else:
        res_art['context'] = context
        res_art['docs'] = docs
        res_art['url'] = art['url']
        with open(f'data/res_data/{arti}.json', 'w', encoding='utf-8') as f:
            json.dump(res_art, f, ensure_ascii=False, indent=4)

