import os
import numpy as np
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI


def call_gpt(ground_truth, prediction):
    client = OpenAI()
    content = f"""Given that we have established matching pairs such as "\'Machine learning\' and \'artificial intelligence\'", 
    "\'Computational Geometry\' and \'Algebraic Geometry\'", "\'Physics and Society\' and \'Physics\'",
    "\'teether\' and \'baby_dental_care\'", "\'earn\' and \'earnings\'", "\'electrical_safety\' and \'electronics_troubleshooting\'", 
    "\'acq\' and \'acquisitions\'", "\'money-fx\' and \'monetary policy\'", when using util.dot_score to measure semantic similarity 
    between tokens, would you consider \'{ground_truth}\' and \'{prediction}\' as a matching pair in a text classification problem? 
    Please respond with \'Yes\' or \'No\'."""
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in text classification, with specialized skills in discerning matching pairs for labels."},
            {"role": "user", "content": content }
        ]
    )
    return str(completion.choices[0].message.content)

def get_ground_truth(args):
    if args.task == 'AAPD':
        file2 = open(f'{args.path}/{args.task}/true_labels.txt', 'r')
        labels = file2.readlines()
        label_dic = {}
        for row in labels:
            key = row.strip().split(';')[0].split('.')[0]
            #key + '.' + 
            label_dic[row.strip().split(';')[0]] = row.strip().split(';')[1][1:]
        true_label = []
        for key in label_dic:
            if not label_dic[key] in true_label:
                true_label.append(label_dic[key])
    elif args.task == 'Reuters-21578':
        file2 = open(f'{args.path}/{args.task}/train_label.txt', 'r')
        labels = file2.readlines()
        true_label = []
        for row in labels:
            label_list = row.strip().split(' ')
            for label in label_list:
                if not label in true_label:
                    true_label.append(label)
    elif args.task == 'RCV1':
        file2 = open(f'{args.path}/{args.task}/rcv1.topics.hier.orig.txt', 'r')
        labels = file2.readlines() 
        true_label = []
        for row in labels[1:]:
            new_label = row.strip().split('child-description: ')[1].lower()
            true_label.append(new_label)
    elif args.task == 'DBPedia-298' or args.task == 'Amazon-531':
        file2 = open(f'{args.path}/{args.task}/train/labels.txt', 'r')
        labels = file2.readlines()
        true_label = []
        for row in labels:
            true_label.append(row.strip().split('\t')[1])
    else:
        raise ValueError('Task not found')

    return true_label



def main(args):
    file = open(f'{args.path}/{args.task}/{args.data_dir}/majority_label_list.txt', 'r')
    majority_labels = file.readlines()
    majority_label_list = []  
    for row in majority_labels:
        labels = []
        maj_list = row.strip().split('\'')
        for i in range(1,len(maj_list)-1, 2):
            labels.append(maj_list[i])
        majority_label_list.append(labels)

    for iteration in range(10, len(majority_labels)):
        file1 = open(f'{args.path}/{args.task}/{args.data_dir}/update_labelspace{iteration+1}.txt', 'r')
        documents = file1.readlines()  
        label_space = []
        for row in documents:
            label = row.strip()
            label_space.append(label)
    
        predict_label_space = label_space + majority_label_list[iteration]
        true_label = get_ground_truth(args)
        ground_truth_length = len(true_label)

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

        passage_embedding = model.encode(predict_label_space)
        sim_matrix = np.empty((0,len(predict_label_space)))
        for i in range(len(true_label)):
            query_embedding = model.encode(true_label[i])
            sim_matrix = np.append(sim_matrix, util.dot_score(query_embedding, passage_embedding).numpy(), 0)
        print(sim_matrix.shape)

        sim_list = []
        for i, sim_score in enumerate(sim_matrix):
            rank_list = np.argsort(sim_score)[-40:]
            cur_list = [true_label[i]]
            for index in rank_list:
                cur_list.append((sim_score[index], predict_label_space[index]))
            if cur_list[-1][0] >= 0.5:
                sim_list.append(cur_list)
        
        pair_list = []
        for line in sim_list:
            true_label = line[0]
            for row in line[1:]:
                #only keep the sim score which is higher than 0.5
                if row[0] >= 0.5:
                    pair_list.append([(true_label, row[1]), row[0]])
        sorted_pair_list = sorted(pair_list, key=lambda x: x[1], reverse = True)

        refine_pair_list = []
        while len(sorted_pair_list) > 0:
            #append the pair and the score
            pair = sorted_pair_list.pop(0)
            score = pair[1]
            ground_truth, prediction = pair[0]

            print(f"Processing pair: {ground_truth} and {prediction} with score {score}")

            if score > 0.75:
                # Delete the pair from the list
                refine_pair_list.append(pair)
                sorted_pair_list = [p for p in sorted_pair_list if ground_truth not in p[0] and prediction not in p[0]]
                continue
            elif 0.5 <= score <= 0.75:
                gpt_response = call_gpt(ground_truth, prediction)
                print(f"GPT Response: {gpt_response}")
                if 'yes' in gpt_response.lower():
                    refine_pair_list.append(pair)
                    sorted_pair_list = [p for p in sorted_pair_list if ground_truth not in p[0] and prediction not in p[0]]
                    continue
        
        with open(f'{args.path}/{args.task}/{args.data_dir}/{args.output_dir}', 'a') as file:
            file.write( '*********' + str(iteration) + '***************' + '\n')

            for item in refine_pair_list:
                file.write(str(item) + '\n')
            file.write(f'Length of the list: {len(refine_pair_list)}, coverage: {len(refine_pair_list)/ground_truth_length} \n')

            file.write('******************************' + '\n')

        # Print the length of the list
        print("Length of the list:", len(refine_pair_list))

        with open(f'{args.path}/{args.task}/{args.data_dir}/iter_coverage_score_new', 'a') as f_score:
            f_score.write(f'Iteration {iteration} coverage: {len(refine_pair_list)/ground_truth_length} \n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/result")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--output_dir", type=str, default="iter_output_pairs_new.txt")
    args = parser.parse_args()

    main(args)