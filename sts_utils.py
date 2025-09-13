

import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub
import os, torch
from bert_score import BERTScorer
from laser_encoders import LaserEncoderPipeline

import numpy as np
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def embed(use_model,input):
    return use_model(input)

def sts(predicted_answer, silver_answer, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder):
    with torch.device("cpu"):
        _, _, stsBERTScoreF1 = scorer.score([silver_answer], [predicted_answer])
    with tf.device('/CPU:0'):
        message_embeddings = embed(use_model,[silver_answer,predicted_answer])
        stsUSE = float(cosine_similarity(message_embeddings[0], message_embeddings[1]))
        pair = tf.constant([silver_answer, predicted_answer])
        embeddings = labse_encoder(labse_preprocessor(pair))["default"]
        stsLaBSE = float(cosine_similarity(embeddings[0], embeddings[1]))
        embeddings = laser_encoder.encode_sentences([silver_answer, predicted_answer])
        stsLaser = float(cosine_similarity(embeddings[0], embeddings[1]))
    return float(stsBERTScoreF1), stsUSE, stsLaBSE, stsLaser

if __name__ == '__main__':
    os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
    os.environ['TFHUB_CACHE_DIR'] = "data/cache_models"

    print('Loading STS models')
    with torch.device("cpu"):
        scorer = BERTScorer(model_type='bert-base-multilingual-cased', device='cpu')
    with tf.device('/CPU:0'):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        use_model = hub.load(module_url)
        labse_preprocessor = hub.KerasLayer("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2")
        labse_encoder = hub.KerasLayer("https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2")
        laser_encoder = LaserEncoderPipeline(laser="laser2")
    
    print('STS models are loaded')
    print('Sentence 1: I am a student.\nSentence 2: I am a teacher.')
    print(sts('I am a student.', 'I am a teacher.', scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder))

