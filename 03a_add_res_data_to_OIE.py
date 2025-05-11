# read res data and add to oie data t24
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
ml = glob.glob('data/gen2oie_data2/*.json')

for fn in tqdm(ml, ncols=100):
    # fn = '/media/data_dump/Ritwik/git/mqna/data/gen2oie_data2/182230.json'
    # print('\n',fn)
    with open(fn, 'r') as f:
        oie_art = json.load(f)
    if 'res_data' in oie_art:
        continue
    # print(oie_art['url'])
    # input('wait')
    # continue
    url = oie_art['url']
    oie_art['res_data'] = []
    try:
        res_art = res_dicts[url]
    except KeyError:
        continue
    cased_words = res_art['docs'][0]['cased_words']
    # print('keys',res_art['docs'][0].keys())
    sent_ids = res_art['docs'][0]['sent_id']
    span_clusters = res_art['docs'][0]['span_clusters']
    for sc in span_clusters:        
        sct = []
        for span in sc:
            span_sentid = sent_ids[span[0]]
            mention = ' '.join(cased_words[span[0]:span[1]])
            if type(oie_art['context']) == str:
                para = oie_art['context'].split('\n')[span_sentid]
            else:
                assert type(oie_art['context']) == list
                para = oie_art['context'][span_sentid]
            mention_context = ' '.join(cased_words[span[0]:span[1]+1])
            para_triples = oie_art['context_triples'][span_sentid]
            if 'gen2oie' in fn:
                para_triples = [para_triples]
            for sti, sent_triples in enumerate(para_triples):
                # if 'gen2oie' in fn:
                #     sent_triples = [sent_triples]
                for ti, triple in enumerate(sent_triples):
                    for ei, entity in enumerate(triple):
                        if mention in entity:
                            # if para.count(mention) > 1:
                            #     triple_words = ' '.join(triple).split()
                            #     print('more than one')
                            #     print(triple_words)
                            #     print(mention_context)
                            #     if len(set(triple_words).intersection(set(mention_context))) > len(mention.split()):
                            #         print('found commons')
                            #         sct.append((span_sentid, sti, ti, ei))
                            # else:
                            #     print('single')
                            #     sct.append((span_sentid, sti, ti, ei))

                            # rule: if mention is exactly == entity then okay, but if entity > mention then entity should have mention_context as well
                            if entity == mention:
                                sct.append((span_sentid, sti, ti, ei))
                            elif len(entity) > len(mention):
                                if mention_context in entity:
                                    sct.append((span_sentid, sti, ti, ei))
                            else:
                                print('mention',mention)
                                print('entity',entity)
                                print('mention_context',mention_context)
                                input('wait')

            # print('mention',mention)
            # print('sct',sct)
            # input('wait')
        oie_art['res_data'].append(sct)
    
    # save the json file
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(oie_art, f, ensure_ascii=False, indent=4)
    # input('wait')


                            
                            
    