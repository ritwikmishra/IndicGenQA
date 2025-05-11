# this code calculates the rouge scores between model output and silver answer t31v3

import argparse, glob, os, re, json, pytz, datetime
from tqdm import tqdm
from rouge_score import rouge_scorer

from indictrans import Transliterator
hin2eng = Transliterator(source='hin', target='eng', build_lookup=True).transform
tam2eng = Transliterator(source='tam', target='eng', build_lookup=True).transform
tel2eng = Transliterator(source='tel', target='eng', build_lookup=True).transform
urd2eng = Transliterator(source='urd', target='eng', build_lookup=True).transform
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=False)
# scorer.score(hin2eng('हामिद अंसारी पूर्व डिप्लोमैट रहे हैं.'),hin2eng('हामिद अंसारी पूर्व डिप्लोमैट भी रहे हैं.'))
# {'rouge1': Score(precision=0.8571428571428571, recall=1.0, fmeasure=0.923076923076923), 'rouge2': Score(precision=0.6666666666666666, recall=0.8, fmeasure=0.7272727272727272), 'rouge3': Score(precision=0.4, recall=0.5, fmeasure=0.4444444444444445), 'rougeL': Score(precision=0.8571428571428571, recall=1.0, fmeasure=0.923076923076923)}

IST = pytz.timezone('Asia/Kolkata') # your time zone here

import wandb
wandb.login(key=input('enter wandb key:'))

id_to_test_file = {
    'gn3': 'data/jsonData2/indie_gen_test_setting0.jsonl',
    'gn5': 'data/jsonData2/indie_gen_test_setting0.jsonl',
    'gn9': 'data/jsonData2/indie_gen_test_setting0.jsonl',
    'gn6': 'data/jsonData2/indie_gen_test_setting1.jsonl',
    'gn7': 'data/jsonData2/indie_gen_test_setting1.jsonl',
    'gn8': 'data/jsonData2/indie_gen_test_setting1.jsonl',
    'gn10': 'data/jsonData2/indie_gen_test_setting2.1.jsonl',
    'gn11': 'data/jsonData2/indie_gen_test_setting2.1.jsonl',
    'gn12': 'data/jsonData2/indie_gen_test_setting2.1.jsonl',
    'gn13': 'data/jsonData2/indie_gen_test_setting2.2.jsonl',
    'gn14': 'data/jsonData2/indie_gen_test_setting2.2.jsonl',
    'gn15': 'data/jsonData2/indie_gen_test_setting2.2.jsonl',
    'gn16': 'data/jsonData2/indie_gen_test_setting3.jsonl',
    'gn17': 'data/jsonData2/indie_gen_test_setting3.jsonl',
    'gn18': 'data/jsonData2/indie_gen_test_setting3.jsonl',
    'gn19': 'data/jsonData2/indie_gen_test_setting4.jsonl',
    'gn20': 'data/jsonData2/indie_gen_test_setting4.jsonl',
    'gn21': 'data/jsonData2/indie_gen_test_setting4.jsonl',
    'gn22': 'data/jsonData2/indie_gen_test_setting5.jsonl',
    'gn23': 'data/jsonData2/indie_gen_test_setting5.jsonl',
    'gn24': 'data/jsonData2/indie_gen_test_setting5.jsonl',
    'gn25': 'data/jsonData2/gen2oie_gen_test_setting1.jsonl',
    'gn26': 'data/jsonData2/gen2oie_gen_test_setting1.jsonl',
    'gn27': 'data/jsonData2/gen2oie_gen_test_setting1.jsonl',
    'gn28': 'data/jsonData2/gen2oie_gen_test_setting2.1.jsonl',
    'gn29': 'data/jsonData2/gen2oie_gen_test_setting2.1.jsonl',
    'gn30': 'data/jsonData2/gen2oie_gen_test_setting2.1.jsonl',
    'gn31': 'data/jsonData2/gen2oie_gen_test_setting3.jsonl',
    'gn32': 'data/jsonData2/gen2oie_gen_test_setting3.jsonl',
    'gn33': 'data/jsonData2/gen2oie_gen_test_setting3.jsonl',
    'gn34': 'data/jsonData2/gen2oie_gen_test_setting4.jsonl',
    'gn35': 'data/jsonData2/gen2oie_gen_test_setting4.jsonl',
    'gn36': 'data/jsonData2/gen2oie_gen_test_setting4.jsonl',
    'gn48': 'data/jsonData2/indie_gen_test_setting6.jsonl',
    'gn49': 'data/jsonData2/indie_gen_test_setting6.jsonl',
    'gn50': 'data/jsonData2/indie_gen_test_setting6.jsonl',
    'gn52': 'data/jsonData2/indie_gen_test_setting5_3.jsonl',
    'gn53': 'data/jsonData2/indie_gen_test_setting5_3.jsonl',
    'gn54': 'data/jsonData2/indie_gen_test_setting5_3.jsonl',
    'gn55': 'data/jsonData2/indie_gen_test_setting5_4.jsonl',
    'gn56': 'data/jsonData2/indie_gen_test_setting5_4.jsonl',
    'gn57': 'data/jsonData2/indie_gen_test_setting5_4.jsonl',
    'gn58': 'data/jsonData2/indie_gen_test_setting5_5.jsonl',
    'gn59': 'data/jsonData2/indie_gen_test_setting5_5.jsonl',
    'gn60': 'data/jsonData2/indie_gen_test_setting5_5.jsonl',
    'gn61': 'data/jsonData2/indie_gen_test_setting5_6.jsonl',
    'gn62': 'data/jsonData2/indie_gen_test_setting5_6.jsonl',
    'gn63': 'data/jsonData2/indie_gen_test_setting5_6.jsonl'
}

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
# parser.add_argument("--test", type=str, help="Show Dataset File")
parser.add_argument('--this_split', type=str, default="1/1")
args = parser.parse_args()
print(datetime.datetime.now(IST).strftime("%c"))
run = wandb.init(project="genmqna2_rouge", id=args.id, name=args.id, resume='allow')

this_split = int(args.this_split.split('/')[0])
total_splits = int(args.this_split.split('/')[1])

test_file_path = id_to_test_file[args.id]
# read jsonl file
test_data = []
with open(test_file_path, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines, ncols=100, desc='Reading Test Data'):
        test_data.append(json.loads(line))

output = "data/jsonData2/out/"+args.id+"/"

bl = [x.split('/')[-1] for x in glob.glob(output+'test_output_base/*.txt')]
ftl = [x.split('/')[-1] for x in glob.glob(output+'test_output_ft/*.txt')]

print('Length of base list:', len(bl))
print('Length of fine-tuned list:', len(ftl))
common = sorted(list(set(bl).intersection(ftl)), key = lambda x: int(x.split('.')[0]))
print('Length of common list:', len(common))

if not os.path.exists(output+'test_rouge'):
    os.makedirs(output+'test_rouge')
if not os.path.exists(output+'test_rouge_chatgpt'):
    os.makedirs(output+'test_rouge_chatgpt')

pbar = tqdm(enumerate(common), ncols=100, desc=args.id+' '+args.this_split+' Comparing', total=len(common))
for filei, c in pbar:
    pbar.set_description(args.id+' '+args.this_split+' Comparing %s' % c)
    if os.path.exists(output+'test_rouge/'+c):
        # open the file
        with open(output+'test_rouge/'+c, 'r') as file:
            result_dict = json.loads(file.read())
        if len(result_dict) == 3:
            continue
        else:
            pass
    if filei % total_splits != this_split-1:
        continue
    with open(output+'test_output_base/'+c, 'r') as file:
        base = file.read()
    with open(output+'test_output_ft/'+c, 'r') as file:
        ft = file.read()
    base_answer = base.split('##Answer')[1].split('##')[0].strip()
    ft_answer = ft.split('##Answer')[1].split('##')[0].strip()
    cnum = int(c.split('.')[0])
    silver_answer = test_data[cnum]['prompt'].split('##Answer')[1].split('##')[0].strip()
    lang = re.search(r'bbc\.(com|co\.\w\w)\/\w+\/', test_data[cnum]['url'])[0].split('/')[1]
    transliterator = {'hindi':hin2eng, 'tamil':tam2eng, 'telugu':tel2eng, 'urdu':urd2eng}[lang]
    baseVsSilver = scorer.score(transliterator(base_answer), transliterator(silver_answer))
    ftVsSilver = scorer.score(transliterator(ft_answer), transliterator(silver_answer))
    baseVsFt = scorer.score(transliterator(base_answer), transliterator(ft_answer))
    # baseVsSilver = sts(base_answer, silver_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    # ftVsSilver = sts(ft_answer, silver_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    # baseVsFt = sts(base_answer, ft_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    result_dict = {'baseVsSilver':baseVsSilver, 'ftVsSilver':ftVsSilver, 'baseVsFt':baseVsFt}
    with open(output+'test_rouge/'+c, 'w') as file:
        file.write(json.dumps(result_dict))
    if args.id in ['gn3','gn22']:
        with open(output+'test_output_gpt-4o-mini/'+c.replace('.txt','.json'), 'r') as file: # /media/data_dump/Ritwik/git/mqna/data/jsonData2/out/gn3/test_output_gpt-4o-mini
            chatgpt_base = json.loads(file.read())
        chatgpt_answer = chatgpt_base['response']
        chatgptVsSilver = scorer.score(transliterator(chatgpt_answer), transliterator(silver_answer))
        chatgptVsFt = scorer.score(transliterator(chatgpt_answer), transliterator(ft_answer))
        result_dict = {'chatgptVsSilver':chatgptVsSilver, 'chatgptVsFt':chatgptVsFt}
        with open(output+'test_rouge_chatgpt/'+c, 'w') as file:
            file.write(json.dumps(result_dict))


