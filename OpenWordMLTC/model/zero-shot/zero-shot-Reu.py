import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

file1 = open('../../../datasets/Reuters-21578/predictLabels/label_user.txt', 'r')
documents = file1.readlines()  

label_space = []
for row in documents:
    label = row.strip()
    label_space.append(label)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

file2 = open('../../../datasets/Reuters-21578/train_texts_split_50.txt', 'r')
docs = file2.readlines()[:4000]

top_labels = []
for doc in docs:
    query_embedding = model.encode(doc)
    passage_embedding = model.encode(label_space)
    sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
    rank_list = np.argsort(sim_scores)[-8:]
    exp_label = []
    for index in rank_list:
        exp_label.append(label_space[index])
    top_labels.append(exp_label)

zstc = pipeline("zero-shot-classification", model="BSC-LT/sciroshot", device = 4)
template = "This example is {}"

with open("../../../datasets/Reuters-21578/zero_shot_text_train.json", 'w') as f:
    for i, doc in enumerate(docs):
        sentence = doc.strip()
        output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
        del output["sequence"]
        print(output)
        json.dump(output, f)
        f.write('\n')