import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from argparse import ArgumentParser
import jsonlines
import os


def label_deletion(major_label_list, text_label_dic, majority_num, max_majority_num = 5):
    for i, label_pair in enumerate(text_label_dic):
        if label_pair[1] > majority_num and i < max_majority_num:
            major_label_list.append(label_pair[0])

    return major_label_list
    
def read_label_space(args):
    file = open(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace.txt', 'r')
    documents = file.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)
    return label_space

def read_cur_label_space(args, index):
    file = open(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace{index}.txt', 'r')
    documents = file.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)
    return label_space

def write_to_label_space(args, label_space):
    if os.path.exists(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace.txt'):
        os.remove(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace.txt')

    with open(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace.txt', 'a') as the_file:
        for label in label_space:
            the_file.write(label + '\n')

def write_to_new_label_space(args, label_space, iter_index):
    with open(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace{iter_index+1}.txt', 'a') as the_file:
        for label in label_space:
            the_file.write(label + '\n')

def zero_shot_training(args, iter_index, file_name, doc_size):
    file1 = open(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace{iter_index +1}.txt', 'r')
    documents = file1.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 'cuda')

    file2 = open(f'{args.path}/{args.task}/{file_name}', 'r')
    docs = file2.readlines()[:doc_size]

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

    zstc = pipeline("zero-shot-classification", model= args.model, device = 'cuda')
    template = "This example is {}"

    with open(f'{args.path}/{args.task}/{args.llama_model}/result/zero_shot_text_train{iter_index+1}.jsonl', 'w') as f:
        for i, doc in enumerate(docs):
            sentence = doc.strip()
            output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
            del output["sequence"]
            print(output)
            json.dump(output, f)
            f.write('\n')

def string_edit(key):
    if key[0] == '[':
        key = key[1:]
    if key[-1] == ']':
        key = key[:-1]
    return key

def self_training(args, iter_index):
    with jsonlines.open(f'{args.path}/{args.task}/{args.llama_model}/result/zero_shot_text_train{iter_index}.jsonl', 'r') as jsonl_f:
        json_raw_list = [obj for obj in jsonl_f] 

    acc_list_raw = np.zeros(len(json_raw_list))
    for i in range(len(json_raw_list)):
        acc_list_raw[i] = json_raw_list[i]['scores'][0]   

    rank_list = np.argsort(acc_list_raw)[:args.tail_set_size]

    file1 = open(f'{args.path}/{args.task}/{args.keyphrase_dir}', 'r')
    keyword_docs = file1.readlines()[:len(json_raw_list)]
    total_word_list = []
    for i, row in enumerate(keyword_docs):
        cur_list = row.strip().split(": ")[1].split(', ')
        total_word_list.extend(cur_list) 

    tail_array = []
    for index in rank_list:
        label = int(keyword_docs[index].split(': ')[0])
        cur_index = index
        row = [index]
        while int(keyword_docs[cur_index - 1].split(': ')[0]) == label:
            cur_index -= 1
            row.append(cur_index)
        cur_index = index
        while (cur_index < (len(keyword_docs)-1)) and (int(keyword_docs[cur_index + 1].split(': ')[0]) == label):
            cur_index += 1
            row.append(cur_index)
        tail_array.append(row)

    select_tail_array = tail_array
    # select_tail_array = []
    # for example in tail_array:
    #     flag = 1
    #     for node in example:
    #         if json_raw_list[node]['scores'][0] > 0.64:
    #             flag = 0
    #     if flag == 1:
    #         select_tail_array.append(example)

    keyword_dic = {}
    keyword_list = []
    for index_list in select_tail_array:
        label_doc = keyword_docs[index_list[0]]
        for keyword in label_doc.strip().split(': ')[1].split(', ')[:3]:
            keyword_list.append(keyword)
            count = 0
            if len(index_list) > 1:
                for index in index_list[1:]:
                    test_label_doc = keyword_docs[index]
                    test_keyword_list = test_label_doc.strip().split(': ')[1].split(', ')
                    if keyword in test_keyword_list:
                        count +=1
        #keyword_dic[index_list[0]] = (keyword,count)
            if keyword in keyword_dic:
                if count > keyword_dic[keyword]:
                    keyword_dic[keyword] = count
            else:
                keyword_dic[keyword] = count

    add_label = []
    for key in keyword_dic:
        count = 0
        for node in total_word_list:
            if string_edit(node) == string_edit(key):
                count +=1
        if count - keyword_dic[key] >= 15:
            add_label.append(string_edit(key))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 'cuda')
    sim_matrix = np.empty((0,len(add_label)))
    for i in range(len(add_label)):
        query_embedding = model.encode(add_label[i])
        passage_embedding = model.encode(add_label)
        sim_matrix = np.append(sim_matrix, util.dot_score(query_embedding, passage_embedding).numpy(), 0)
    
    sim_list = []
    deleted_list = []
    for i, sim_score in enumerate(sim_matrix):
        for j in range(len(sim_score)):
            if sim_score[j] > (args.sim_threshold + iter_index / 100)  and sim_score[j] < 1.1:
                if i < j:
                    sim = [i, add_label[i], j , add_label[j], sim_score[j]]
                    sim_list.append(sim)
                    if not add_label[i] in deleted_list:
                        deleted_list.append(add_label[i])
    for label in deleted_list:
        add_label.remove(label)

    predict_label_space = read_label_space(args)

    passage_embedding = model.encode(predict_label_space)
    final_sim_matrix = np.empty((0,len(predict_label_space)))
    for i in range(len(add_label)):
        query_embedding = model.encode(add_label[i])
        final_sim_matrix = np.append(final_sim_matrix, util.dot_score(query_embedding, passage_embedding).numpy(), 0)

    final_sim_list = []
    for i, sim_score in enumerate(final_sim_matrix):
        rank_list = np.argsort(sim_score)[-3:]
        cur_list = [add_label[i]]
        for index in rank_list:
            cur_list.append((sim_score[index], predict_label_space[index]))
        final_sim_list.append(cur_list)

    final_add_label = []
    for row in final_sim_list:
        if row[3][0] < (args.sim_threshold + iter_index / 100) :
            final_add_label.append(row[0])
    return final_add_label[:args.max_add_label]

    

def main(args):
    print(args.tail_set_size, args.majority_num, args.max_majority_num, args.sim_threshold, args.max_add_label)
    major_label_list = []
    for iter_index in range(10):

        with jsonlines.open(f'{args.path}/{args.task}/{args.llama_model}/result/zero_shot_text_train{iter_index}.jsonl', 'r') as jsonl_f:
            json_raw_list = [obj for obj in jsonl_f]

        acc_list_raw = np.zeros(len(json_raw_list))
        label_list_raw = []
        for i in range(len(json_raw_list)):
            acc_list_raw[i] = json_raw_list[i]['scores'][0]
            label_list_raw.append(json_raw_list[i]['labels'][0]) 
        document_size = len(json_raw_list)

        #new label space
        label_space = read_label_space(args)
        cur_label_space = read_cur_label_space(args, iter_index)

        text_label_dic = {item: 0 for item in cur_label_space}
        for label in label_list_raw:
            text_label_dic[label] += 1

        # keyword_sorted_dic = sorted(keyword_label_dic.items(), key=lambda x:x[1], reverse = True)
        text_sorted_dic = sorted(text_label_dic.items(), key=lambda x:x[1], reverse = True)            

        #delete minority label
        remove_key_list = []
        for key in text_label_dic:
            if (text_label_dic[key] <= 6) and (key in label_space):
                print(key)
                remove_key_list.append(key)
                label_space.remove(key)
        
        write_to_label_space(args, label_space)

        with open(f'{args.path}/{args.task}/{args.llama_model}/result/remove_label_list.txt', 'a') as the_file:
            the_file.write(str(remove_key_list) + '\n')

        final_add_label = self_training(args, iter_index)

        with open(f'{args.path}/{args.task}/{args.llama_model}/result/add_label_list.txt', 'a') as the_file:
            the_file.write(str(final_add_label) + '\n')

        with open(f'{args.path}/{args.task}/{args.llama_model}/result/update_labelspace.txt', 'a') as the_file:
            for label in final_add_label:
                the_file.write(label + '\n')
        print(final_add_label)

        new_label_space = label_space + final_add_label

        #delete majority label
        major_label_list = label_deletion(major_label_list, text_sorted_dic, args.majority_num, args.max_majority_num)
        for label in major_label_list:
            if label in new_label_space:
                new_label_space.remove(label)
        write_to_new_label_space(args, new_label_space, iter_index)

        with open(f'{args.path}/{args.task}/{args.llama_model}/result/majority_label_list.txt', 'a') as the_file:
            the_file.write(str(major_label_list) + '\n')

        print(major_label_list)        

        zero_shot_training(args, iter_index, args.data_dir, document_size)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="llama2_label_50.txt")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--llama_model", type=str, default='llama2')
    parser.add_argument("--tail_set_size", type=int, default=500, choices=[500, 1000, 1500])
    parser.add_argument("--majority_num", type=int, default=350, choices=[350, 400, 500, 650])
    parser.add_argument("--max_majority_num", type=int, default=5, choices=[5, 10])
    parser.add_argument("--sim_threshold", type=float, default=0.55, choices=[0.55, 0.60, 0.65])
    parser.add_argument("--max_add_label", type=int, default=10, choices=[10, 15, 20])
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    args = parser.parse_args()

    main(args)