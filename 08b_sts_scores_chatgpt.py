# this code calculates the sts scores for chatgpt outputs on gn3 and gn22 t31v2

import argparse, glob, os, json, pytz, datetime
from tqdm import tqdm
from sts_utils import *

IST = pytz.timezone('Asia/Kolkata') # your time zone here

import wandb
wandb.login(key=input('enter wandb key:'))

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
os.environ['TFHUB_CACHE_DIR'] = "data/cache_models"

print('Loading STS models')
with torch.device("cuda:0"): # 'cuda:0' 'cpu'
    scorer = BERTScorer(model_type='bert-base-multilingual-cased', device='cpu')
with tf.device('/GPU:0'): # '/GPU:0' '/CPU:0'
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2
    # https://tfhub.dev/google/universal-sentence-encoder/4
    use_model = hub.load(module_url)
    labse_preprocessor = hub.KerasLayer("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2")
    labse_encoder = hub.KerasLayer("https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2")
    laser_encoder = LaserEncoderPipeline(laser="laser2")

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
    'gn60': 'data/jsonData2/indie_gen_test_setting5_5.jsonl'
}

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, help="Show Model")
# parser.add_argument("--test", type=str, help="Show Dataset File")
parser.add_argument('--this_split', type=str, default="1/1")
args = parser.parse_args()
print(datetime.datetime.now(IST).strftime("%c"))
run = wandb.init(project="genmqna2_sts", id=args.id, name=args.id, resume='allow')

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

if not os.path.exists(output+'test_sts_chatgpt'):
    os.makedirs(output+'test_sts_chatgpt')

pbar = tqdm(enumerate(common), ncols=100, desc=args.id+' '+args.this_split+' Comparing', total=len(common))
for filei, c in pbar:
    pbar.set_description(args.id+' '+args.this_split+' Comparing %s' % c)
    if os.path.exists(output+'test_sts_chatgpt/'+c):
        continue
    if filei % total_splits != this_split-1:
        continue
    try:
        with open(output+'test_output_gpt-4o-mini/'+c.replace('.txt','.json'), 'r') as file: # /media/data_dump/Ritwik/git/mqna/data/jsonData2/out/gn3/test_output_gpt-4o-mini
            base = json.loads(file.read())
        with open(output+'test_output_ft/'+c, 'r') as file:
            ft = file.read()
    except Exception as e:
        print(e)
        print(c)
        continue
    base_answer = base['response']
    ft_answer = ft.split('##Answer')[1].split('##')[0].strip()
    cnum = int(c.split('.')[0])
    silver_answer = base['prompt'].split('\n\n##Answer\n')[1].split('##')[0].strip()
    # print('\nbase_answer',base_answer)
    # print('ft_answer', ft_answer)
    # print('silver_answer',silver_answer)
    # input('wait')
    
    baseVsSilver = sts(base_answer, silver_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    # ftVsSilver = sts(ft_answer, silver_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    baseVsFt = sts(base_answer, ft_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    result_dict = {'baseVsSilver':baseVsSilver, 'baseVsFt':baseVsFt}
    with open(output+'test_sts_chatgpt/'+c, 'w') as file:
        file.write(json.dumps(result_dict))



