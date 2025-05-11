# create prompts from oie data t29

import argparse, re, pickle
import json, glob, heapq
from tqdm import tqdm
from collections import defaultdict

def short_url(url):
    return re.search(r'www\.bbc\.(com|co\.uk)/.+', url)[0].split('?')[0]

def top_k_indices(lst, k):
    # Use heapq.nlargest to get the k largest values along with their indices
    return [i for i, _ in heapq.nlargest(k, enumerate(lst), key=lambda x: x[1])]

parser = argparse.ArgumentParser()
parser.add_argument("--oie", type=str, default='indie')
parser.add_argument("--setting", type=str)
args = parser.parse_args()

# settings
# 0: Q + context <------ same for both oie
# 1: Q + all triples 
# 2: Q + top-k triples as per APS
#   2.1    each
#   2.2    sent
# 3: Q + all coref chain paras
# 4: Q + top-k coref chain paras as per APS
# 5: Q + top-k paras as per APS <------ same for both oie
# 6: Q + top-k paras as per APS and coref <------ same for both oie

with open('data/q_a_dict.pkl', 'rb') as f:
    q_a_dict = pickle.load(f)

# read train, val, test urls
with open('data/train_urls_gen.pkl', 'rb') as f:
    train_urls = [short_url(u) for u in pickle.load(f)]
with open('data/val_urls_gen.pkl', 'rb') as f:
    val_urls = [short_url(u) for u in pickle.load(f)]
with open('data/test_urls_gen.pkl', 'rb') as f:
    test_urls = [short_url(u) for u in pickle.load(f)]

all_urls = train_urls + val_urls + test_urls
print('len(all_urls):', len(all_urls))
all_urls_dict = defaultdict(bool)
for url in all_urls:
    all_urls_dict[url] = True

ml = glob.glob('data/'+args.oie+'_data2/*.json')

if args.setting == '0':
    jsonlines_list = []
    for fname in tqdm(ml, ncols=100, total=len(ml)):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        context = ' '.join(art['context']) if type(art['context']) == list else art['context'].replace('\n', ' ')

        for qid in art['qids']:
            ans = q_a_dict[su+'|'+qid]
            que = art['q_coref_chain_para_scores_2'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{context}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))
    with open('data/jsonData2/'+args.oie+'_gen_train_setting0.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting0.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting0.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '1':
    jsonlines_list = []
    for fname in tqdm(ml, ncols=100):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        triples = art['context_triples']
        flat_triples = []
        for l1 in triples:
            for l2 in l1:
                if 'gen2oie' in fname:
                    flat_triples.append(' '.join(l2))
                else:
                    for l3 in l2:
                        flat_triples.append(' '.join(l3))
        triple_text = '. '.join(flat_triples)

        for qid in art['qids']:
            ans = q_a_dict[su+'|'+qid]
            que = art['q_coref_chain_para_scores_2'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{triple_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting1.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting1.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting1.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '2.1':
    topk = 10
    jsonlines_list = []
    for fname in tqdm(ml, ncols=100):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        triples = art['context_triples']
        flat_triples = []
        for l1 in triples:
            for l2 in l1:
                if 'gen2oie' in fname:
                    flat_triples.append(' '.join(l2))
                else:
                    for l3 in l2:
                        flat_triples.append(' '.join(l3))

        for qid in art['qids']:
            try:
                assert len(art['q_context_each_2_triple_scores'][qid][1:]) == len(flat_triples)
            except:
                print(fname)
                print(len(art['q_context_each_2_triple_scores'][qid][1:]), len(flat_triples))
                exit()
            topk_triples = top_k_indices(art['q_context_each_2_triple_scores'][qid][1:], topk)
            triple_text = []
            for k in topk_triples:
                triple_text.append(flat_triples[k])
            triple_text = '. '.join(triple_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_coref_chain_para_scores_2'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{triple_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting2.1.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting2.1.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting2.1.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '2.2':
    topk = 10
    jsonlines_list = []
    for fname in tqdm(ml, ncols=100):
        assert 'gen2oie' not in fname
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        triples = art['context_triples']
        flat_triples = []
        for l1 in triples:
            for l2 in l1:
                if 'gen2oie' in fname:
                    flat_triples.append(' '.join(l2))
                else:
                    sent_triples = []
                    for l3 in l2:
                        sent_triples.append(' '.join(l3))
                    flat_triples.append('. '.join(sent_triples))

        for qid in art['qids']:
            topk_triples = top_k_indices(art['q_context_sent_2_triple_scores'][qid][1:], topk)
            triple_text = []
            for k in topk_triples:
                triple_text.append(flat_triples[k])
            triple_text = '. '.join(triple_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_coref_chain_para_scores'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{triple_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting2.2.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting2.2.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting2.2.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '3':
    jsonlines_list = []
    for fname in tqdm(ml, ncols=100):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        paras = art['coref_chain_paras']
        flat_paras = []
        for l1 in paras:
            for l2 in l1:
                flat_paras.append(' '.join(l2))
        para_text = '. '.join(flat_paras)

        for qid in art['qids']:
            ans = q_a_dict[su+'|'+qid]
            que = art['q_coref_chain_para_scores_2'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting3.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting3.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting3.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')     

elif args.setting == '4':
    topk = 5
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        coref_chain_paras = art['coref_chain_paras']

        for qid in art['qids']:
            assert len(art['q_coref_chain_para_scores_2'][qid][1:]) == len(coref_chain_paras)
            topk_paras = top_k_indices(art['q_coref_chain_para_scores_2'][qid][1:], topk)
            para_text = []
            for k in topk_paras:
                para_text.append(' '.join(coref_chain_paras[k]))
            para_text = '. '.join(para_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_coref_chain_para_scores_2'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
        
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting4.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting4.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting4.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '5':
    topk = 5
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        context_paras = art['context'] if type(art['context']) == list else art['context'].split('\n')

        for qid in art['qids']:
            assert len(art['q_context_para_scores_with_prev'][qid][1:]) == len(context_paras)
            topk_paras = top_k_indices(art['q_context_para_scores_with_prev'][qid][1:], topk)
            para_text = []
            for k in topk_paras:
                para_text.append(context_paras[k])
            para_text = '. '.join(para_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_context_para_scores_with_prev'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting5.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting5.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting5.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '5_3':
    topk = 5
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        context_paras = art['context'] if type(art['context']) == list else art['context'].split('\n')

        for qid in art['qids']:
            assert len(art['q_context_para_scores_2_tydi_munfquad'][qid][1:]) == len(context_paras)
            topk_paras = top_k_indices(art['q_context_para_scores_2_tydi_munfquad'][qid][1:], topk)
            para_text = []
            for k in topk_paras:
                para_text.append(context_paras[k])
            para_text = '. '.join(para_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_context_para_scores_2_tydi_munfquad'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting5_3.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting5_3.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting5_3.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '5_4':
    topk = 5
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        context_paras = art['context'] if type(art['context']) == list else art['context'].split('\n')

        for qid in art['qids']:
            assert len(art['q_context_para_scores_2_only_tydi'][qid][1:]) == len(context_paras)
            topk_paras = top_k_indices(art['q_context_para_scores_2_only_tydi'][qid][1:], topk)
            para_text = []
            for k in topk_paras:
                para_text.append(context_paras[k])
            para_text = '. '.join(para_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_context_para_scores_2_only_tydi'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting5_4.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting5_4.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting5_4.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '5_5':
    ml = glob.glob('data/lchain_test_results/setting5_5/*.json')
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        for qas in art:
            su = short_url(qas['url'])
            if not all_urls_dict[su]:
                continue
            qid = qas['qid']
            que = qas['question']
            para_text = qas['retreived_docs']
            ans = qas['answer']
            assert ans == q_a_dict[su+'|'+qid]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': qas['url']})
    
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting5_5.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting5_5.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting5_5.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

elif args.setting == '5_6':
    topk = 5
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        context_paras = art['context'] if type(art['context']) == list else art['context'].split('\n')

        for qid in art['qids']:
            assert len(art['q_context_para_scores_bm25'][qid][1:]) == len(context_paras)
            topk_paras = top_k_indices(art['q_context_para_scores_bm25'][qid][1:], topk)
            para_text = []
            for k in topk_paras:
                para_text.append(context_paras[k])
            para_text = '. '.join(para_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_context_para_scores_2'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
        
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting5_6.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting5_6.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting5_6.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')


elif args.setting == '6':
    topk = 5
    jsonlines_list = []
    for fname in tqdm(ml, ncols=80):
        with open(fname, 'r') as f:
            art = json.load(f)
        su = short_url(art['url'])
        if not all_urls_dict[su]:
            continue
        context_paras = art['context'] if type(art['context']) == list else art['context'].split('\n')

        for qid in art['qids']:
            assert len(art['q_context_para_scores_with_prev'][qid][1:]) == len(context_paras)
            top_para_index = top_k_indices(art['q_context_para_scores_with_prev'][qid][1:], 1)[0]
            topk_paras = []
            for para_cluster in art['para_res_data']:
                if top_para_index in para_cluster:
                    topk_paras += para_cluster
            if len(topk_paras) > 0:
                assert max(topk_paras) < len(context_paras)
                topk_paras = [(x,art['q_context_para_scores_with_prev'][qid][1:][x]) for x in list(set(topk_paras))]
                topk_paras = sorted(topk_paras, key=lambda x: x[1], reverse=True)[:topk]
                topk_paras = [x[0] for x in topk_paras]
            else:
                topk_paras = [top_para_index]
            para_text = []
            for k in topk_paras:
                para_text.append(context_paras[k])
            para_text = '. '.join(para_text)
            ans = q_a_dict[su+'|'+qid]
            que = art['q_context_para_scores_with_prev'][qid][0]
            prompt = f'Answer the question based on the given context.\n\n##Question\n{que}\n\n##Context\n{para_text}\n\n##Answer\n{ans}'
            jsonlines_list.append({'prompt': prompt, 'qid': qid, 'url': art['url']})
            
    train_jsonlines, val_jsonlines, test_jsonlines = [], [], []
    for item in jsonlines_list:
        su = short_url(item['url'])
        if su in train_urls:
            train_jsonlines.append(item)
        elif su in val_urls:
            val_jsonlines.append(item)
        else:
            test_jsonlines.append(item)
    print('len(jsonlines_list):', len(jsonlines_list))
    print('len(train_jsonlines):', len(train_jsonlines))
    print('len(val_jsonlines):', len(val_jsonlines))
    print('len(test_jsonlines):', len(test_jsonlines))

    with open('data/jsonData2/'+args.oie+'_gen_train_setting6.jsonl', 'w') as f:
        for item in tqdm(train_jsonlines, ncols=100, desc='train'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_val_setting6.jsonl', 'w') as f:
        for item in tqdm(val_jsonlines, ncols=100, desc='val'):
            json.dump(item, f)
            f.write('\n')
    with open('data/jsonData2/'+args.oie+'_gen_test_setting6.jsonl', 'w') as f:
        for item in tqdm(test_jsonlines, ncols=100, desc='test'):
            json.dump(item, f)
            f.write('\n')

else:
    print('Error')

