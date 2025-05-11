# this code runs langchain retriever on munfquad testset t37
# and creates the prompt files with them

# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from t_utils import *
# from torch.utils import data
import importlib, pickle, hashlib
from tqdm import tqdm
# import bs4, torch, 
import json, os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from collections import defaultdict
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="arguments here")
parser.add_argument("--s", default="5")
parser.add_argument("--r", default="yes")
# parser.add_argument("--c", default="no")
# parser.add_argument("--q", default="not Quantized")
parser.add_argument('--this_split', type=str, default="1/1")
args = parser.parse_args()

this_split = int(args.this_split.split('/')[0])
total_splits = int(args.this_split.split('/')[1])

model_path = "data/cache_models"
model_id = "google/gemma-2b-it"

# from transformers import BitsAndBytesConfig
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# if args.q == "Quantized":
#     double_quant_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_type=torch.float16)
#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config, trust_remote_code=True, device_map="auto", cache_dir=model_path)
# else:
#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", cache_dir=model_path)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500, torch_dtype=torch.bfloat16,trust_remote_code=True,device_map="auto",num_return_sequences=1, do_sample=True, top_k=1, temperature=0.005)
# llm = HuggingFacePipeline(pipeline=pipe)

from langchain_community.document_transformers import LongContextReorder
reordering = LongContextReorder()

def format_docs(docs):
    if args.r == 'yes':
        docs = reordering.transform_documents(docs)
    return docs

# from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# prompt3 = ChatPromptTemplate(input_variables=['context', 'question'], 
#                              metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, 
#                              messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], 
#                                                                                         template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
             

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

print('Loading dataset')
with open('data/dataset/MuNfQuAD_v2.pkl','rb') as f:
    data = pickle.load(f)
print('dataset loaded')

# create paths if they do not exist
if not os.path.exists('data/lchain_test_results/setting'+args.s):
    os.makedirs('data/lchain_test_results/setting'+args.s)

# write description to file
# with open('data/lchain_test_results/setting'+args.s+'/aa_desc.txt', 'w') as f:
#     f.write('setting'+args.s+' '+args.r+' reordering '+args.c+' compression_retriever '+args.q+' llm\n')

test_data = []

for datum in tqdm(data, total=len(data), desc='loading test data', ncols=100):
    if all_urls_dict[short_url(datum['url'])]:
        test_data.append(datum)

print('test data loaded')



del data

for arti, item in tqdm(enumerate(test_data), total=len(test_data), desc='testset', ncols=100):
    if arti % total_splits != this_split - 1:
        continue
    if os.path.exists('data/lchain_test_results/setting'+args.s+'/'+str(arti)+'.json'):
        continue
    test_results = []
    url = item['url']
    context = item['paragraphs'][0]['context']
    single_doc = [Document(page_content=context)]
    splits = []
    for p in context.split('\n'):
        splits.append(Document(page_content=p))
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2", cache_folder=model_path))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    for qas in item['paragraphs'][0]['qas']:
        question = qas['question']
        qid = qas['id']
        retrieved_docs = retriever.get_relevant_documents(question)
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"Document {i+1}:")
        #     print(doc.page_content)
        # input('wait')
        re_ordered_retrieved_docs = format_docs(retrieved_docs)
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"Document {i+1}:")
        #     print(doc.page_content)
        # input('wait')
        # exit()
        retrieved_docs = '\n'.join([doc.page_content for doc in retrieved_docs])
        re_ordered_retrieved_docs = '\n'.join([doc.page_content for doc in re_ordered_retrieved_docs])
        resulti = {'url':url,'question':question, 'qid':qas['id'], 'retreived_docs':retrieved_docs, 're_ordered_retrieved_docs':re_ordered_retrieved_docs,'answer':qas['answers'][0]['text']}
        test_results.append(resulti)
    with open('data/lchain_test_results/setting'+args.s+'/'+str(arti)+'.json', 'w', encoding='utf8') as f:
        json.dump(test_results, f, ensure_ascii=False)
        
    vectorstore.delete_collection()

# with open('data/lchain_test_results/setting'+args.s+'/aa_data.json', 'w', encoding='utf8') as f:
#     json.dump(test_results, f, indent=4, ensure_ascii=False)



