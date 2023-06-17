import json
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast
import pandas as pd
import numpy as np
import random
import torch
import csv
import sys


input_file = sys.argv[1]
output_file = sys.argv[2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(420)

tag_to_id = {
    'B-company': 0,
    'B-facility': 1,
    'B-geo-loc': 2,
    'B-movie': 3,
    'B-musicartist': 4,
    'B-other': 5,
    'B-person': 6,
    'B-product': 7,
    'B-sportsteam': 8,
    'B-tvshow': 9,
    'I-company': 10,
    'I-facility': 11,
    'I-geo-loc': 12,
    'I-movie': 13,
    'I-musicartist': 14,
    'I-other': 15,
    'I-person': 16,
    'I-product': 17,
    'I-sportsteam': 18,
    'I-tvshow': 19,
    'O': 20
}


# Create id2label and label2id mappings
# Create id2label and label2id mappings
id2label = {v: k for k, v in tag_to_id.items()}
label2id = tag_to_id


#Load and update the model configuration
# config = json.load(open("twitter_ner/config.json"))
# config["id2label"] = id2label
# config["label2id"] = label2id
# json.dump(config, open("twitter_ner/config.json", "w"))

config = json.load(open("twitter_ner_2/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("twitter_ner_2/config.json", "w"))

#Load the model and tokenizer
# model = AutoModelForTokenClassification.from_pretrained("twitter_ner")
# tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

model = AutoModelForTokenClassification.from_pretrained("twitter_ner_2")
tokenizer = BertTokenizerFast.from_pretrained("tokenizer_ner_2")


def load_sentences_from_csv(filepath):
    test_data = pd.read_csv(filepath, skip_blank_lines=False)
    sentences = []
    sentence = []
    for _, row in test_data.iterrows():
        if pd.isna(row['word']):  #Check for nan
            #print(sentence)
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(row['word'])
    if sentence:  #last sentence if not empty
        sentences.append(sentence)


    sentences = [' '.join(sentence) for sentence in sentences]

    return sentences

def tokenize_and_get_word_ids(sentence): 
    new_sentence = sentence.split()
    tokenized_inputs = tokenizer(new_sentence, truncation=True, is_split_into_words=True) 
    word_ids = tokenized_inputs.word_ids()

    # Assign None to subtokens
    previous_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id == previous_word_id:
            word_ids[i] = None
        previous_word_id = word_id

    return tokenized_inputs, word_ids

sentences = load_sentences_from_csv(input_file)
master_id = 0

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])  
    for sentence in sentences:
        inputs, word_ids = tokenize_and_get_word_ids(sentence)
        #print(word_ids)
        inputs = {k: torch.tensor([v]) for k, v in inputs.items()}  # Convert to tensors

        #Get model's predictions
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

        #Map the predictions back to the original words
        for word_id, prediction in zip(word_ids, predictions):
            if word_id is not None:  
                writer.writerow([master_id, prediction])
                master_id += 1