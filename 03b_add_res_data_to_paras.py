# this code reads res data and add it to paras (instead of oie triples, it deals with context paras) t24v2

import json, glob, pickle, os
from tqdm import tqdm    

# # open res data
# ml = glob.glob('data/res_data/*.json')

# res_dicts = {}

# for fn in tqdm(ml, ncols=100):
#     with open(fn, 'r') as f:
#         res_art = json.load(f)
#     url = res_art['url']
#     res_dicts[url] = res_art

# # save pickle
# with open('data/res_data/res_dicts.pkl','wb') as f:
#     pickle.dump(res_dicts, f)
# exit()

print('loading pickle')
## load pickle
with open('data/res_data/res_dicts.pkl','rb') as f:
    res_dicts = pickle.load(f)
print('pickle loaded')    

# open oie data
ml = glob.glob('data/indie_data2/*.json')

for fn in tqdm(ml, ncols=100):
    with open(fn, 'r') as f:
        oie_art = json.load(f)
    if 'para_res_data' in oie_art:
        continue
    
    url = oie_art['url']
    oie_art['para_res_data'] = []
    try:
        res_art = res_dicts[url]
    except KeyError:
        continue
    sent_ids = res_art['docs'][0]['sent_id']
    span_clusters = res_art['docs'][0]['span_clusters']
    for sc in span_clusters:
        sct = []
        for span in sc:
            span_sentid = sent_ids[span[0]]
            if span_sentid not in sct:
                sct.append(span_sentid)
        if sct not in oie_art['para_res_data']:
            oie_art['para_res_data'].append(sct)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(oie_art, f, ensure_ascii=False, indent=4)
