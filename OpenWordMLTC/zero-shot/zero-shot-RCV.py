import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json

def zero_shot_training(file_name, cur_type):
    file1 = open(f'../../datasets/RCV1/no_human_result/update_labelspace.txt', 'r')
    documents = file1.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 3)

    file2 = open(f'../../datasets/RCV1/{file_name}_50.txt', 'r')
    docs = file2.readlines()[:14125]

    top_labels = []
    for i, doc in enumerate(docs):
        print(i)
        query_embedding = model.encode(doc)
        passage_embedding = model.encode(label_space)
        sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
        rank_list = np.argsort(sim_scores)[-8:]
        exp_label = []
        for index in rank_list:
            exp_label.append(label_space[index])
        top_labels.append(exp_label)

    print("pass")

    zstc = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", device = 3)
    template = "This example is {}"

    with open(f'../../datasets/RCV1/no_human_result/zero_shot_{cur_type}_train0.jsonl', 'w') as f:
        for i, doc in enumerate(docs):
            sentence = doc.strip()
            output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
            del output["sequence"]
            print(output)
            json.dump(output, f)
            f.write('\n')

def main():
    zero_shot_training('llama_label', 'keyword')
    zero_shot_training('train_texts_split', 'text')

if __name__ == "__main__":
    main()