# CUDA_VISIBLE_DEVICES=1 python hi_predict.py --weight data/xlmr2.pt --output_file out_xlmr_2.jsonlines --experiment xlmr
import argparse

import jsonlines
import torch, time
from tqdm import tqdm

from coref import CorefModel
from coref.tokenizer_customization import *

import stanza, os
import importlib, re
stanza_path = importlib.util.find_spec('stanza').submodule_search_locations[0]
# https://stackoverflow.com/questions/269795/how-do-i-find-the-location-of-python-module-sources
# https://stackoverflow.com/questions/35288021/what-is-the-equivalent-of-imp-find-module-in-importlib
stanza_version = stanza.__version__

if not os.path.exists(stanza_path+'/'+stanza_version+'/'):
    os.makedirs(stanza_path+'/'+stanza_version+'/')

def build_doc(doc: dict, model: CorefModel) -> dict:
    filter_func = TOKENIZER_FILTERS.get(model.config.bert_model,
                                        lambda _: True)
    token_map = TOKENIZER_MAPS.get(model.config.bert_model, {})

    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (token_map[word]
                          if word in token_map
                          else model.tokenizer.tokenize(word))
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id

    doc["head2span"] = []
    if "speaker" not in doc:
        doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []

    return doc

def resolve_coreferences(text_list, model):
    json_list = []
    json_object = {"document_id": "mu", "cased_words": [], "sent_id":[], "pos":[]}
    snum = -1
    for sent in text_list:
        snum+=1
        for word in sent.split():
            json_object['cased_words'].append(word)
            json_object['sent_id'].append(snum)
    json_list.append(json_object)
    
    docs = [build_doc(json_object, model) for json_object in json_list]

    with torch.no_grad():
        for doc in tqdm(docs, unit="docs", disable=True):
            a = time.time()
            result = model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters

            for key in ("word2subword", "subwords", "word_id", "head2span","speaker"):
                del doc[key]
            
            # doc['cased_words2'] = [str(i)+'_'+x for i,x in enumerate(doc['cased_words'])]
            # doc['pos2'] = [str(i)+'_'+x for i,x in enumerate(doc['pos'])]
    
    return docs

    

if __name__ == "__main__":
    # span clusters means these word-spans are clustered according to their predicted coreference chains
    # word_clusters means those head_words which belong to one chain, similar to span_clusters but at word level
    # remember both are 0 indexed
    # and in span clusters [a,b] means a is included whereas b is not included

    # distribution of PRONs in UD data
    # Counter({'उन्होंने': 1509, 'इसके': 703, 'यह': 684, 'अपने': 627, 'वह': 618, 'कोई': 591, 'उनके': 490, 'उन्हें': 468, 'अब': 464, 'अपनी': 437, 'किसी': 427, 'वे': 338, 'उनकी': 335, 'इससे': 294, 'जब': 257, 'यहां': 257, 'इस': 251, 'जो': 234, 'अभी': 222, 'उसके': 213, 'यहाँ': 212, 'उसे': 209, 'इसमें': 193, 'इसे': 178, 'हम': 168, 'इसका': 164, 'वहां': 164, 'अपना': 162, 'उनका': 162, 'इसकी': 149, 'तो': 141, 'इसलिए': 124, 'जिसमें': 123, 'कुछ': 118, 'उसने': 118, 'उसकी': 110, 'क्या': 103, 'तब': 96, 'खुद': 92, 'मैं': 88, 'इनमें': 87, 'ऐसा': 85, 'आप': 79, 'जिससे': 78, 'हमें': 75, 'जहां': 74, 'उसका': 69, 'कभी': 60, 'उनसे': 59, 'हमारे': 57, 'जिसके': 56, 'सभी': 56, 'जिसे': 55, 'उससे': 53, 'ऐसे': 52, 'वहाँ': 48, 'मुझे': 46, 'हमने': 45, 'कैसे': 41, 'हमारी': 41, 'क्यों': 40, 'एक': 39, 'दूसरे': 39, 'उस': 38, 'इन्हें': 37, 'मैंने': 36, 'जहाँ': 35, 'ये': 35, 'जिन्हें': 33, 'जिनमें': 32, 'जिसका': 31, 'इसी': 29, 'उन': 29, 'इनकी': 27, 'इनके': 26, 'उसमें': 26, 'सब': 26, 'उनमें': 24, 'तभी': 24, 'हमारा': 22, 'आपको': 21, 'जिसकी': 21, 'मेरे': 21, 'कब': 19, 'जिनके': 19, 'जिस': 17, 'आपके': 16, 'इनका': 16, 'यही': 16, 'मेरी': 16, 'कौन': 15, 'जिसने': 15, 'उनको': 15, 'यहीं': 14, 'जिन्होंने': 14, 'इसीलिए': 13, 'वही': 13, 'आपकी': 12, 'मेरा': 12, 'वैसे': 11, 'कहाँ': 11, 'वहीं': 11, 'कहीं': 11, 'उसी': 10, 'इसको': 10, 'जिनकी': 10, 'इन': 9, 'इनसे': 8, 'स्वयं': 8, 'आपस': 8, 'सबसे': 7, 'आपका': 7, 'जिन': 7, 'जिनका': 7, 'आपने': 6, 'सबके': 6, 'जिनसे': 6, 'किसने': 5, 'इन्हीं': 4, 'इन्होंने': 4, 'वो': 4, 'इनको': 4, 'ऐसी': 4, 'किस': 4, 'कहां': 4, 'जिसको': 4, 'मुझ': 4, 'परस्पर': 4, 'कितना': 3, 'हमसे': 3, 'मुझसे': 3, 'जैसा': 3, 'इसने': 3, 'सबको': 3, 'उन्\u200dहोंने': 2, 'जोकि': 2, 'सबकी': 2, 'किसके': 2, 'कैसा': 2, 'किसे': 2, 'जिनको': 2, 'उसको': 2, 'तुमने': 2, 'उन्\u200dहें': 1, 'कैसी': 1, 'मैने': 1, 'स्वतः': 1, 'किन': 1, 'इतना': 1, 'तुम्हारे': 1, 'कितने': 1, 'और': 1, 'जैसे': 1, 'कितनों': 1, 'सबका': 1, 'किसका': 1, 'हमको': 1, 'उन्होनें': 1, 'तहां': 1, 'यूं': 1, 'किसकी': 1, 'उन्हीं': 1, 'कई': 1, 'हममें': 1, 'अन्य': 1, 'इन्\u200dहीं': 1, 'यत्र': 1, 'तत्र': 1, 'वैसा': 1})

    model = CorefModel("coref/config.toml", "xlmr")
    model.load_weights(path="/media/data_dump/Ritwik/git/wl-coref/data2/xlmr_multi_plus_hi2.pt", map_location="cpu",ignore={"bert_optimizer", "general_optimizer","bert_scheduler", "general_scheduler"})
    model.training = False

    raw_sentences = 'हमने कोहरे में एक आदमी को भागते देखा। उसने काले रंग की शाल पहनी थी। उसका कद लम्बा था।'.split('।')
    # raw_sentences = ['मार्च 2001 में एक बार फिर अमरीका के अट्ठाइसवें रक्षा सचिव के रूप में उनकी वापसी हुई .']

    docs = resolve_coreferences(raw_sentences, model)
    print(docs)
    exit()
    assert len(raw_sentences) == len(out_sentences)
    for x,y in zip(raw_sentences, out_sentences):
        print('In ', x)
        print('Out', y, end='\n\n')
    
    raw_sentences = ["Ramchandra was a good king. His brother's name was Laxman.",'رام چندر جی اچھے بادشاہ تھے۔ اس کے بھائی کا نام لکشمن تھا۔',"ராமச்சந்திரா ஒரு நல்ல அரசர். அவரது சகோதரரின் பெயர் லக்ஷ்மன்.","రామచంద్ర మంచి రాజు. అతని సోదరుడి పేరు లక్ష్మణ్.","രാമചന്ദ്രജി ഒരു നല്ല രാജാവായിരുന്നു. സഹോദരന്റെ പേര് ലക്ഷ്മണൻ.","রামচন্দ্র জি একজন ভালো রাজা ছিলেন। তার ভাইয়ের নাম ছিল লক্ষ্মণ।","રામચંદ્રજી સારા રાજા હતા. તેમના ભાઈનું નામ લક્ષ્મણ હતું.","ৰামচন্দ্ৰ জী এজন ভাল ৰজা আছিল। ভায়েকৰ নাম লক্ষ্মণ।","ਰਾਮਚੰਦਰ ਜੀ ਚੰਗੇ ਰਾਜੇ ਸਨ। ਉਸ ਦੇ ਭਰਾ ਦਾ ਨਾਂ ਲਕਸ਼ਮਣ ਸੀ।","रामचंद्र जी इक अच्छे राजा थे। उंदे भ्राऽ दा नां लक्ष्मण हा।"]
    docs, out_sentences = resolve_coreferences_for_non_hindi(raw_sentences, model)
    assert len(raw_sentences) == len(out_sentences)
    for x,y in zip(raw_sentences, out_sentences):
        print('In ', x)
        print('Out', y, end='\n\n')


# Examples
# In  श्री राम विष्णु जी के अवतार थे। उस अवतार में उन्होंने लंकापति रावण का वध किया। वह एक कुशल योद्धा थे। मर्यादा पुरषोत्तम कहे जाने वाले राम एक अच्छे राजा भी थे। उन्होंने अपना राज त्याग दिया। उनके भाई का नाम लक्ष्मण था और उनके पिता का नाम दशरथ था। सीता की खीज में वे लंका पहुंचे। उन्हें वहां सीता मिल गयी। लंका एक टापू देश था। वह चारो ओर से पानी से घिरा था। उन्होंने वहां रावण का वध किया। 
# Out श्री राम विष्णु जी के अवतार थे । उस अवतार में श्री राम ने लंकापति रावण का वध किया । श्री राम एक कुशल योद्धा थे । मर्यादा पुरषोत्तम कहे जाने वाले राम एक अच्छे राजा भी थे । श्री राम ने अपना राज त्याग दिया । श्री राम के भाई का नाम लक्ष्मण था और श्री राम के पिता का नाम दशरथ था । सीता की खीज में श्री राम लंका पहुंचे । श्री राम को वहां सीता मिल गयी । लंका एक टापू देश था । लंका चारो ओर से पानी से घिरा था । श्री राम ने वहां रावण का वध किया ।

# In  हमने कोहरे में एक आदमी को भागते देखा।  उसने काले रंग की शाल पहनी थी।  उसका कद लम्बा था। 
# Out हमने कोहरे में एक आदमी को भागते देखा । आदमी ने काले रंग की शाल पहनी थी । आदमी का कद लम्बा था ।

# In  हमने कोहरे में दो आदमियों को भागते देखा।  उन्होंने काले रंग की शाल पहनी थी।  उनका कद लम्बा था। 
# Out हमने कोहरे में दो आदमियों को भागते देखा । दो आदमियों ने काले रंग की शाल पहनी थी । दो आदमियों का कद लम्बा था ।

# In  कल रात को दो दोस्त पुल्ल पे बैठे थे। उनमे से एक पुल्ल से गिर गया। वह तैरकर किनारे पे आया। उसका दोस्त भागकर उसके पास आया। 
# Out कल रात को दो दोस्त पुल्ल पे बैठे थे । दो दोस्त से एक पुल्ल से गिर गया । वह तैरकर किनारे पे आया । उसका दोस्त भागकर उसका दोस्त के पास आया ।

# In  Ramchandra was a good king. His brother's name was Laxman.
# Out Ramchandra was a good king. Ramchandra brother's name was Laxman.

# In  رام چندر جی اچھے بادشاہ تھے۔ اس کے بھائی کا نام لکشمن تھا۔
# Out رام چندر جی اچھے بادشاہ تھے۔ رام چندر جی بھائی اس کے نام لکشمن تھا۔

# In  ராமச்சந்திரா ஒரு நல்ல அரசர். அவரது சகோதரரின் பெயர் லக்ஷ்மன்.
# Out ராமச்சந்திரா ஒரு நல்ல அரசர். ராமச்சந்திரா சகோதரரின் பெயர் லக்ஷ்மன்.

# In  రామచంద్ర మంచి రాజు. అతని సోదరుడి పేరు లక్ష్మణ్.
# Out రామచంద్ర మంచి రాజు. రామచంద్ర సోదరుడి పేరు లక్ష్మణ్.

# In  രാമചന്ദ്രജി ഒരു നല്ല രാജാവായിരുന്നു. സഹോദരന്റെ പേര് ലക്ഷ്മണൻ.
# Out രാമചന്ദ്രജി ഒരു നല്ല രാജാവായിരുന്നു. സഹോദരന്റെ പേര് ലക്ഷ്മണൻ.

# In  রামচন্দ্র জি একজন ভালো রাজা ছিলেন। তার ভাইয়ের নাম ছিল লক্ষ্মণ।
# Out রামচন্দ্র জি একজন ভালো রাজা ছিলেন। রামচন্দ্র জি ভাইয়ের নাম ছিল লক্ষ্মণ।

# In  રામચંદ્રજી સારા રાજા હતા. તેમના ભાઈનું નામ લક્ષ્મણ હતું.
# Out રામચંદ્રજી સારા રાજા હતા. રામચંદ્રજી ભાઈનું નામ લક્ષ્મણ હતું.

# In  ৰামচন্দ্ৰ জী এজন ভাল ৰজা আছিল। ভায়েকৰ নাম লক্ষ্মণ।
# Out ৰামচন্দ্ৰ জী এজন ভাল ৰজা আছিল। ভায়েকৰ নাম লক্ষ্মণ।

# In  ਰਾਮਚੰਦਰ ਜੀ ਚੰਗੇ ਰਾਜੇ ਸਨ। ਉਸ ਦੇ ਭਰਾ ਦਾ ਨਾਂ ਲਕਸ਼ਮਣ ਸੀ।
# Out ਰਾਮਚੰਦਰ ਜੀ ਚੰਗੇ ਰਾਜੇ ਸਨ। ਰਾਮਚੰਦਰ ਜੀ ਭਰਾ ਦਾ ਨਾਂ ਲਕਸ਼ਮਣ ਸੀ।

# In  रामचंद्र जी इक अच्छे राजा थे। उंदे भ्राऽ दा नां लक्ष्मण हा।
# Out रामचंद्र जी इक अच्छे राजा थे। उंदे भ्राऽ दा नां लक्ष्मण हा।

