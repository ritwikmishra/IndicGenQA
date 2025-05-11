# find intersection between indie and gen2oie data t23_v2
# create url splits

import pickle, glob, json, re, os
from tqdm import tqdm
from collections import defaultdict
import random
random.seed(123)

def short_url(url):
    return re.search(r'www\.bbc\.(com|co\.uk)/.+', url)[0].split('?')[0]

print('Reading training urls')
with open('data/train_urls.pkl','rb') as f:
    train_urls = pickle.load(f)

train_urls = {short_url(url):True for url in train_urls}
train_urls = defaultdict(lambda: False, train_urls)

ml = glob.glob('data/indie_data2/*.json')
# ml2 = glob.glob('data/res_data/*.json')
ml3 = glob.glob('data/gen2oie_data2/*.json')

gen2oie_urls = []
for fname in tqdm(ml3, ncols=100, desc='gen2oie'):
    with open(fname, 'r') as f:
        oie_art = json.load(f)
    url = oie_art['url']
    su = short_url(url)
    if not train_urls[su]:
        gen2oie_urls.append(url)

indie_urls = []
for fname in tqdm(ml, ncols=100, desc='indie'):
    with open(fname, 'r') as f:
        indie_art = json.load(f)
    url = indie_art['url']
    su = short_url(url)
    if not train_urls[su]:
        indie_urls.append(url)

print('len:', len(indie_urls), len(gen2oie_urls))

# intersection
commons = list(set(indie_urls).intersection(set(gen2oie_urls)))

print('common len:', len(commons))

# randomly split
random.shuffle(commons)

train_urls_for_gen = commons[:int(0.7*len(commons))]
val_urls_for_gen = commons[int(0.7*len(commons)):int(0.85*len(commons))]
test_urls_for_gen = commons[int(0.85*len(commons)):]

print('train:', len(train_urls_for_gen))
print('val:', len(val_urls_for_gen))
print('test:', len(test_urls_for_gen))

# write to file
with open('data/train_urls_gen.pkl', 'wb') as f:
    pickle.dump(train_urls_for_gen, f)

with open('data/val_urls_gen.pkl', 'wb') as f:
    pickle.dump(val_urls_for_gen, f)

with open('data/test_urls_gen.pkl', 'wb') as f:
    pickle.dump(test_urls_for_gen, f)

