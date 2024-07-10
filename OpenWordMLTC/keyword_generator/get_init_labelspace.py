import os
import numpy as np
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

def main(args):
    file = open(f'{args.path}/{args.task}/{args.data_dir}', 'r')
    documents = file.readlines()
    org_class = []
    for row in documents:
        #label = row.strip().split(": ")[1]
        label = row.split(';')[0]
        print(label)
        #label = row.replace(';', '').strip()
        if not label in org_class:
            org_class.append(label)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sim_matrix = np.empty((0,len(org_class)))
    for i in range(len(org_class)):
        query_embedding = model.encode(org_class[i])
        passage_embedding = model.encode(org_class)
        sim_matrix = np.append(sim_matrix, util.dot_score(query_embedding, passage_embedding).numpy(), 0)
    
    sim_list = []
    for i, sim_score in enumerate(sim_matrix):
        for j in range(len(sim_score)):
            if sim_score[j] > args.lower_bound and sim_score[j] < 0.99:
                if i < j:
                    sim = [i, org_class[i], j , org_class[j], sim_score[j]]
                    sim_list.append(sim)

    client = OpenAI()
    answers = []
    for label_candidate in sim_list:
        content = f'Do labels "{label_candidate[1]}" and "{label_candidate[3]}" have similar meanings in general that we only need one of them to put into the label space?. Please only answer Yes or No. If answer is Yes, please delete the low-level label and only output the label that we should delete with the format "label".'
        completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": content }
            ]
        )
        result = completion.choices[0].message.content
        answers.append([label_candidate[1], label_candidate[3], label_candidate[4], result])
        print(label_candidate[1], label_candidate[3], label_candidate[4], result)

    redundent_labels = []
    for i, answer in enumerate(answers):
        if answer[3].find('Yes') != -1:
            redundent_labels.append(answer)
    delete_list = []
    while len(redundent_labels) > 0:
        redundent = redundent_labels[0]
        delete_label = redundent[3].split('Yes')[1].replace('.', '').replace('\n', '').replace(',', '')
        if delete_label.find('"') != -1:
            delete_label = delete_label.split('"')[1]
        else:
            delete_label = delete_label.split(' ')[1]
        print(delete_label)
        redundent_labels = [label for label in redundent_labels if (label[0] != delete_label and label[1] != delete_label)]
        delete_list.append(delete_label)
    
    for info in delete_list:
        if info in org_class:
            org_class.remove(info)
    

    with open(f'{args.path}/{args.task}/{args.output_dir}', 'a') as the_file:
        for label in org_class:
            the_file.write(label + '\n')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/init_labelspace.txt")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--lower_bound", type=int, default=0.80)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--output_dir", type=str, default="llama2/init_label_space.txt")
    args = parser.parse_args()

    main(args)