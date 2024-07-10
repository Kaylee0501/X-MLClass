import os
from argparse import ArgumentParser
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json



def zero_shot_training(args, file_name, cur_type):
    file1 = open(f'{args.path}/{args.task}/llama2/init_label_space.txt', 'r')
    documents = file1.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

    file2 = open(f'{args.path}/{args.task}/{file_name}', 'r')
    docs = file2.readlines()

    if cur_type == 'keyword':
        keyphrase_subset = []
        for i, doc in enumerate(docs):
            if int(doc.split(": ")[0]) < args.dynamic_iter:
                keyphrase_subset.append(doc)
            else:
                chunk_size = i
                break
    else:
        keyphrase_subset = []
        for i, doc in enumerate(docs):
            if int(doc.split(" ")[0]) < args.dynamic_iter:
                keyphrase_subset.append(doc)
            else:
                chunk_size = i
                break

    print(len(keyphrase_subset))

    top_labels = []
    for i, doc in enumerate(keyphrase_subset):
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

    zstc = pipeline("zero-shot-classification", model=args.model, device = 'cuda')
    template = "This example is {}"

    path = f'{args.path}/{args.task}/llama2/result'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/zero_shot_{cur_type}_train_0.jsonl', 'w') as f:
        for i, doc in enumerate(keyphrase_subset):
            sentence = doc.strip()
            output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
            del output["sequence"]
            print(output)
            json.dump(output, f)
            f.write('\n')


def main(args):
    zero_shot_training(args, args.keyphrase_dir, 'keyword')
    zero_shot_training(args, args.data_dir, 'text')


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="llama2_label_50.txt")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--dynamic_iter", type=int, default=3000)
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    args = parser.parse_args()

    main(args)