

# run Gen2OIE on MuNfQuAD data t18

import argparse, json, pickle
from tqdm import tqdm
import time, math, re, glob
from collections import defaultdict

parser = argparse.ArgumentParser(description='Gen2OIE on the entire data')
parser.add_argument('--lang', type=str)
parser.add_argument('--this_split', type=str)
args = parser.parse_args()

this_split = int(args.this_split.split('/')[0])
total_splits = int(args.this_split.split('/')[1])

def short_url(url):
    return re.search(r'www\.bbc\.(com|co\.uk)/.+', url)[0].split('?')[0]

print('Loading dataset')
with open('data/dataset/MuNfQuAD_v2.pkl','rb') as f:
    data = pickle.load(f)

print('Train urls loading')
with open('data/train_urls.pkl','rb') as f:
    train_urls = pickle.load(f)

#convert it into default dictionary
train_url_dict = defaultdict(bool)

for tu in tqdm(train_urls, ncols=100, desc='prep'):
    short_url_ = short_url(tu)
    train_url_dict[short_url_] = True

# print('Reading training urls')
# with open('data/train_urls.pkl','rb') as f:
#     train_urls = pickle.load(f)
# # convert train)urls to default dictionary
# train_urls = {short_url(url):True for url in train_urls}
# from collections import defaultdict
# train_urls = defaultdict(lambda: False, train_urls)

language = args.lang
for language in ['hi','te','pt','en']:
    if language == 'hi':
        lang_name = ['hindi','urdu']
    elif language == 'te':
        lang_name = ['telugu','tamil']
    elif language == 'pt':
        lang_name = ['portuguese']
    else:
        lang_name = ['news']

    device = 'cuda:0'

    # sentences = 'मेरठ समेत पश्चिमी उत्तर प्रदेश के 14 ज़िलों में शुरू की गई नई सुविधा के तहत बिजली उपभोक्ता घर बैठे बिल ऑनलाइन माध्यम से जमा कर सकेंगे। बिजली विभाग के अधिकारी आशुतोष निरंजन ने बताया है कि उपभोक्ता अपना बिजली बिल डेबिट-क्रेडिट कार्ड, यूपीआई, मोबाइल वॉलेट और अन्य ऑनलाइन पेमेंट के तरीकों से जमा कर सकते हैं।'.split('।')

    import os
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    s1_path = f'data/gen2oie_models/aact_moie_gen2oie_{language}/stage1/'
    s1_model = AutoModelForSeq2SeqLM.from_pretrained(s1_path).to(device)

    s2_path = f'data/gen2oie_models/aact_moie_gen2oie_{language}/stage2/'
    s2_model = AutoModelForSeq2SeqLM.from_pretrained(s2_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

    def convert_output(all_outputs):
        all_outputs2 = []
        for sentence, extractions in all_outputs:
            extractions2 = []
            for extraction in extractions:
                head = re.findall(r'<a1>(.*?)</a1>', extraction)
                relation = re.findall(r'<r>(.*?)</r>', extraction)
                tail = re.findall(r'<a2>(.*?)</a2>', extraction)
                if len(head) == 0 or len(relation) == 0 or len(tail) == 0:
                    continue
                else:
                    head, relation, tail = head[0].strip(), relation[0].strip(), tail[0].strip()
                    extractions2.append([head, relation, tail])
            all_outputs2.append(extractions2)
        return all_outputs2


    def extract_triples(sentences):
        s1_results = []
        batch_size = 32
        num_batches = math.ceil(len(sentences)/batch_size)
        # print('Stage-1 prediction:')
        for k in tqdm(range(num_batches), disable=True):
            sentences_k = sentences[k*batch_size:(k+1)*batch_size]
            s1_inputs_toks = tokenizer(sentences_k, return_tensors="pt", padding=True)
            s1_inputs_toks = s1_inputs_toks['input_ids'].to(device)
            preds_k = s1_model.generate(s1_inputs_toks, max_length=50, num_beams=1, early_stopping=True)    
            s1_results_k = tokenizer.batch_decode(preds_k, skip_special_tokens=True)
            s1_results.extend(s1_results_k)

        all_outputs = []
        s2_inputs, s2_input_sentences = [], []
        num_relations = []
        nrs = 0
        num_relations.append(0)
        for sentence_i, sentence in enumerate(sentences):
            for rel in s1_results[sentence_i].split('<r>'):
                s2_input = rel.strip() + ' <r> ' + sentence
                s2_inputs.append(s2_input)
                nrs += 1
            num_relations.append(nrs)

        s2_results = []
        batch_size = 8
        num_batches = math.ceil(len(s2_inputs)/batch_size)
        for k in tqdm(range(num_batches), disable=True):
            s2_input_k = s2_inputs[k*batch_size:(k+1)*batch_size]
            s2_inputs_toks = tokenizer(s2_input_k, return_tensors="pt", padding=True)
            s2_inputs_toks = s2_inputs_toks['input_ids'].to(device)

            preds_k = s2_model.generate(s2_inputs_toks, max_length=128, num_beams=1, early_stopping=True)
            s2_results_k = tokenizer.batch_decode(preds_k, skip_special_tokens=True)
            s2_results.extend(s2_results_k)

        s2r = []
        for ri, result in enumerate(s2_results):
            if ri in num_relations:
                s2r.append([result])
            else:
                s2r[-1].append(result)    

        for sentence, s2_result in zip(sentences, s2r):
            def flatten(t):
                return [item for sublist in t for item in sublist if item.strip()]
            extractions = []
            for e1 in s2_result:
                for e2 in e1.split('<e>'): 
                    if e2.strip():
                        extractions.append(e2)
            extractions = set(extractions)
            all_outputs.append([sentence, extractions])
        
        all_outputs = convert_output(all_outputs)
        return all_outputs


    # all_outputs = extract_triples(sentences)

    # for sentence, extractions in all_outputs:
    #     print('Sentence: ')
    #     print(sentence)
    #     print('Extractions: ')
    #     extractions = [str(x) for x in extractions]
    #     print('\n'.join(extractions))  
    #     print('\n')

    # input('wait')
    # exit()

    pbar = tqdm(enumerate(data), ncols=100, total=len(data), desc='Gen2OIE '+str(lang_name))
    done = 0
    for arti,art in pbar:
        if train_url_dict[short_url(art['url'])]:
            continue
        time.sleep(0.0000001)
        found = False
        for ln in lang_name:
            if re.search(r'bbc\.(com|co\.uk)/'+ln+r'/', art['url']):
                found = True
                break
        if not found:
            continue
        if arti%total_splits != this_split-1:
            continue
        if os.path.exists(f'data/gen2oie_data2/{arti}.json'):
            try:
                with open(f'data/gen2oie_data2/{arti}.json', 'r') as f:
                    res_art = json.load(f)
                assert res_art['url'] == art['url'] and res_art['context'] == art['paragraphs'][0]['context'].split('\n')
            except Exception as e:
                print('\n')
                print(res_art['url'])
                print(art['url'])
                print(res_art['context_sents'].keys())
                print(art['paragraphs'][0].keys())
                input('wait')
                pass
            else:
                continue
        ml3 = glob.glob('data/gen2oie_data2/*.json')
        pbar.set_description(f'Gen2OIE {language}: Standing at {arti} Done {done} L {len(ml3)}')
        ind_art = {'url':'', 'context':'', 'context_triples':[], 'qids':[], 'qtriples':[]}
        
        context_sents = art['paragraphs'][0]['context'].split('\n')
        try:
            pbar.set_description(f'Gen2OIE {language}: Processing {arti} Done {done} L {len(ml3)}')
            context_triples = extract_triples(context_sents)
            for qas in art['paragraphs'][0]['qas']:
                question = qas['question']
                question_triples = extract_triples([question])
                ind_art['qids'].append(qas['id'])
                ind_art['qtriples'].append(question_triples)
        except Exception as e:
            print(e)
            # if 'CUDA out of memory' in str(e):
            #     continue
            # else:
            #     raise e
        else:
            ind_art['url'] = art['url']
            ind_art['context'] = context_sents
            ind_art['context_triples'] = context_triples

            with open(f'data/gen2oie_data2/{arti}.json', 'w', encoding='utf-8') as f:
                json.dump(ind_art, f, ensure_ascii=False, indent=4)
            done+=1