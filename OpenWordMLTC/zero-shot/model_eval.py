import json
import numpy as np
import jsonlines
import copy
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os

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

def run_test(args, model_type, array_size):
    with jsonlines.open(f'{args.path}/{args.task}/{args.data_dir}/zero_shot_keyword_test_{model_type}.jsonl', 'r') as jsonl_f:
        json_list = [obj for obj in jsonl_f]
    with jsonlines.open(f'{args.path}/{args.task}/{args.data_dir}/zero_shot_text_test_{model_type}.jsonl', 'r') as jsonl_f:
        json_raw_list = [obj for obj in jsonl_f]

    file1 = open(f'{args.path}/{args.task}/{args.keyphrase_dir}', 'r')
    keyword_docs = file1.readlines()

    count = 0
    for line in keyword_docs:
        if int(line.split(': ')[0]) == 0:
            count +=1
        else:
            break

    array_index_list = []
    iter_index = 0
    new_array = []
    for ct in range(count):
        new_array.append(ct)
        
    for i, row in enumerate(keyword_docs[count:]):
        index = int(row.strip().split(': ')[0])
        if index != iter_index:
            array_index_list.append((iter_index, new_array))
            new_array = [i+count]
            iter_index = index
        else:
            new_array.append(i + count)
    array_index_list.append((iter_index, new_array))

    filtered_index = []
    for line in array_index_list[:array_size]:
        filtered_index.append(line[0])

    merge_json_dic = {}
    for line in array_index_list[:array_size]:
        merge_json_dic[line[0]] = []
        for index in line[1]:
            merge_json_dic[line[0]].append(json_list[index])

    merge_json_raw_dic = {}
    for line in array_index_list[:array_size]:
        merge_json_raw_dic[line[0]] = []
        for index in line[1]:
            merge_json_raw_dic[line[0]].append(json_raw_list[index])

    final_label_list = []
    for iteration, key in enumerate(merge_json_dic):
        print(iteration)
        label_list = []
        keyword_rank_list = merge_json_dic[key]
        text_rank_list = merge_json_raw_dic[key]
        sum_rank_list = keyword_rank_list + text_rank_list
        index_list = []
        for i in range(len(sum_rank_list)):
            index_list.append(i)
        for i in range(8):
            rank_dic = {}
            score_dic = {}
            for index in index_list:
                label = sum_rank_list[index]['labels'][i]
                score = sum_rank_list[index]['scores'][i]
                if not label in rank_dic:
                    rank_dic[label] = 1
                    score_dic[label] = score
                else:
                    rank_dic[label] += 1
                    if score > score_dic[label]:
                        score_dic[label] = score
            sorted_rank_dic = sorted(rank_dic.items(), key=lambda x:x[1], reverse = True)
            if sorted_rank_dic[0][1] > 1:
                best_key_list = [sorted_rank_dic[0][0]]
                best_num = sorted_rank_dic[0][1]
                for new_pair in sorted_rank_dic[1:]:
                    if new_pair[1] == best_num:
                        best_key_list.append(new_pair[0])
                    else:
                        break
                new_score_list = {}
                for best_key in best_key_list:
                    new_score_list[best_key] = score_dic[best_key]
                sorted_score_list = sorted(new_score_list.items(), key=lambda x:x[1], reverse = True)
                for key in sorted_score_list:
                    if not key[0] in label_list:
                        label_list.append(key[0])
            else:
                sorted_score_list = sorted(score_dic.items(), key=lambda x:x[1], reverse = True)
                best_key_list = [sorted_score_list[0][0]]
                if not sorted_score_list[0][0] in label_list:
                    label_list.append(sorted_score_list[0][0])
        final_label_list.append(label_list)

    file = open(f'{args.path}/{args.task}/test_label.txt', 'r')
    documents = file.readlines() 

    true_label_array = []
    if args.task == 'AAPD':
        for index in filtered_index:
            true_label_list = []
            label_list = documents[index][:-3].split('; ')
            for label in label_list:
                new_label = label.split('.')[1]
                if not new_label in true_label_list:
                    true_label_list.append(new_label)
            true_label_array.append(true_label_list)
    else:
        for index in filtered_index:
            true_label_list = []
            if args.task == 'Amazon-531' or 'DBPedia-298':
                label_list = documents[index][:-3].split(', ')
            elif args.task == 'Reuters-21578':
                label_list = documents[index][:-1].split(' ')
            elif args.task == 'RCV1':
                label_list = documents[index][:-3].split('; ')
            for label in label_list:
                new_label = label
                if not new_label in true_label_list:
                    true_label_list.append(new_label)
            true_label_array.append(true_label_list)

    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

    
    prob_array1 = np.zeros(array_size)
    for i in range(array_size):
        predict_labels = final_label_list[i][:1]
        true_labels = copy.deepcopy(true_label_array[i])
        #print(predict_labels)
        #print(true_labels)
        total_size = min(len(true_labels),1)
        count = 0
        for pred_label in predict_labels:
            if len(true_labels) > 0:
                query_embedding = model.encode(pred_label)
                passage_embedding = model.encode(true_labels)
                sim_score = util.dot_score(query_embedding, passage_embedding).numpy()[0]
                #print(pred_label, sim_score)
                rank_list = np.argsort(sim_score)
                if sim_score[rank_list[-1]] >= 0.75:
                    count += 1
                    true_labels.pop(rank_list[-1])
                elif sim_score[rank_list[-1]] >= 0.5:
                    if call_gpt(true_labels[rank_list[-1]], pred_label) == 'Yes':
                        count += 1
                        true_labels.pop(rank_list[-1])
        print(i, count)
        prob_array1[i] = count / total_size 

    answer1 = np.sum(prob_array1) / array_size

    prob_array3 = np.zeros(array_size)
    for i in range(array_size):
        predict_labels = final_label_list[i][:3]
        true_labels = copy.deepcopy(true_label_array[i])
        #print(predict_labels)
        #print(true_labels)
        total_size = min(len(true_labels),3)
        count = 0
        for pred_label in predict_labels:
            if len(true_labels) > 0:
                query_embedding = model.encode(pred_label)
                passage_embedding = model.encode(true_labels)
                sim_score = util.dot_score(query_embedding, passage_embedding).numpy()[0]
                #print(pred_label, sim_score)
                rank_list = np.argsort(sim_score)
                if sim_score[rank_list[-1]] >= 0.75:
                    count += 1
                    true_labels.pop(rank_list[-1])
                elif sim_score[rank_list[-1]] >= 0.5:
                    if call_gpt(true_labels[rank_list[-1]], pred_label) == 'Yes':
                        count += 1
                        true_labels.pop(rank_list[-1])
        print(i, count)
        prob_array3[i] = count / total_size 

    answer3 = np.sum(prob_array3) / array_size
    
    with open(f'{args.path}/{args.task}/{args.output_dir}', 'a') as the_file:
         the_file.write(f'{model_type} 1 {answer1} \n')
         the_file.write(f'{model_type} 3 {answer3} \n')


def main(args):

    run_test(args, 'deberta', args.test_size)
    run_test(args, 'bart', args.test_size)
    run_test(args, 'xlm', args.test_size)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/test_performance")
    parser.add_argument("--keyphrase_dir", type=str, default="llama2_label_test_50.txt")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="llama2/test_performance/MLClass_result.txt")
    args = parser.parse_args()

    main(args)