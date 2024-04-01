import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import jsonlines
import os


def label_deletion(major_label_list, text_label_dic, keyword_label_dic):
    for key in text_label_dic:
        if text_label_dic[key] > 650:
            major_label_list.append(key)
    for key in keyword_label_dic:
        if (keyword_label_dic[key] > 650) and (not key in major_label_list):
            major_label_list.append(key)
    return major_label_list
    
def read_label_space():
    file = open('../datasets/Amazon-531/llm_cluster_result/update_labelspace.txt', 'r')
    documents = file.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)
    return label_space

def write_to_label_space(label_space):
    if os.path.exists("../datasets/Amazon-531/llm_cluster_result/update_labelspace.txt"):
        os.remove("../datasets/Amazon-531/llm_cluster_result/update_labelspace.txt")

    with open('../datasets/Amazon-531/llm_cluster_result/update_labelspace.txt', 'a') as the_file:
        for label in label_space:
            the_file.write(label + '\n')

def write_to_new_label_space(label_space, iter_index):
    with open(f'../datasets/Amazon-531/llm_cluster_result/update_labelspace{iter_index+1}.txt', 'a') as the_file:
        for label in label_space:
            the_file.write(label + '\n')

def zero_shot_training(iter_index, file_name, cur_type):
    file1 = open(f'../datasets/Amazon-531/llm_cluster_result/update_labelspace{iter_index +1}.txt', 'r')
    documents = file1.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 4)

    file2 = open(f'../datasets/Amazon-531/{file_name}_50.txt', 'r')
    docs = file2.readlines()[:31918]

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

    zstc = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", device = 4)
    template = "This example is {}"

    with open(f'../datasets/Amazon-531/llm_cluster_result/zero_shot_{cur_type}_train{iter_index+1}.jsonl', 'w') as f:
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

def self_training(iter_index ):
    with jsonlines.open(f'../datasets/Amazon-531/llm_cluster_result/zero_shot_keyword_train{iter_index+1}.jsonl', 'r') as jsonl_f:
        json_list = [obj for obj in jsonl_f]

    with jsonlines.open(f'../datasets/Amazon-531/llm_cluster_result/zero_shot_text_train{iter_index+1}.jsonl', 'r') as jsonl_f:
        json_raw_list = [obj for obj in jsonl_f] 

    acc_list_keyword = np.zeros(len(json_list))
    acc_list_raw = np.zeros(len(json_raw_list))
    for i in range(len(json_list)):
        acc_list_keyword[i] = json_list[i]['scores'][0]
        acc_list_raw[i] = json_raw_list[i]['scores'][0]   

    rank_list = np.argsort(acc_list_keyword)[:1500]
    super_rank_list = []
    for index in rank_list:
        if acc_list_raw[index] < 0.55:
            super_rank_list.append(index)

    file1 = open('../datasets/Amazon-531/llama_label_50.txt', 'r')
    keyword_docs = file1.readlines()[:31918]
    total_word_list = []
    for row in keyword_docs:
        cur_list = row.strip().split(": ")[1].split(', ')
        total_word_list.extend(cur_list) 

    tail_array = []
    for index in super_rank_list:
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

    select_tail_array = []
    for example in tail_array:
        flag = 1
        for node in example:
            if json_list[node]['scores'][0] > 0.64:
                flag = 0
        if flag == 1:
            select_tail_array.append(example)

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

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 4)
    sim_matrix = np.empty((0,len(add_label)))
    for i in range(len(add_label)):
        query_embedding = model.encode(add_label[i])
        passage_embedding = model.encode(add_label)
        sim_matrix = np.append(sim_matrix, util.dot_score(query_embedding, passage_embedding).numpy(), 0)
    
    sim_list = []
    deleted_list = []
    for i, sim_score in enumerate(sim_matrix):
        for j in range(len(sim_score)):
            if sim_score[j] > 0.55 and sim_score[j] < 1.1:
                if i < j:
                    sim = [i, add_label[i], j , add_label[j], sim_score[j]]
                    sim_list.append(sim)
                    if not add_label[i] in deleted_list:
                        deleted_list.append(add_label[i])
    for label in deleted_list:
        add_label.remove(label)

    predict_label_space = read_label_space()

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
        if row[3][0] < 0.55:
            final_add_label.append(row[0])
    return final_add_label

    

def main():
    major_label_list = ['supplements', 'grooming', 'food_beverages', 'toys_games', 'reviews', 'beauty_products', 'family_kids_toys', 'games', 'toys_gifts', 'health_supplements', 'pets', 'beauty_hair_care', 'product_review', 'beauty_personal_care', 'health_personal_care', 'products_for_children', 'household_supplies', 'pet_supplies', 'pet_care', '[consumer_products_personal_care_shaving]', 'skincare', 'toys_building_sets', 'parenting_toys', 'toys_crafts', 'pet_accessories', 'fragrances', 'pet_care_value', 'shopping_beauty', 'parenting_baby_care', 'health_wellness', 'beauty_skincare', 'educational_toys', 'consumer_electronics', 'health_medical_supplies', 'consumer_complaints', 'home_furniture', 'pet_grooming', 'health_medical_equipment', 'snacks', 'personal_care_grooming', '[grooming_shaving]', 'toys_puzzles', 'toys_action_figures', 'lifestyle_diet_nutrition', 'personal_care_skincare', 'pet_toys', 'negative_review', 'customer_service', 'household_cleaning', 'games_family', 'fashion_beauty', 'parenting_baby_gear', 'parenting_supplies', 'recipes', 'food_snacks', 'consumer_electronics_reviews', 'toys_construction', 'toys_and_games_issues', 'health_pain_management', 'beauty_makeup_tools', 'beauty_makeup', 'food_ingredients', 'supplements_digestive_health', 'product_reviews_negative', 'fashion_beauty_product_issues', 'gift_ideas', 'board_games', 'toys_dolls', 'home_kitchen', 'pet_litter_boxes', 'family_care', 'parenting_infants']
    for iter_index in range(4, 10):
        with jsonlines.open(f'../datasets/Amazon-531/llm_cluster_result/zero_shot_keyword_train{iter_index}.jsonl', 'r') as jsonl_f:
            json_list = [obj for obj in jsonl_f]

        with jsonlines.open(f'../datasets/Amazon-531/llm_cluster_result/zero_shot_text_train{iter_index}.jsonl', 'r') as jsonl_f:
            json_raw_list = [obj for obj in jsonl_f]

        acc_list_keyword = np.zeros(len(json_list))
        label_list_keyword = []
        acc_list_raw = np.zeros(len(json_raw_list))
        label_list_raw = []
        for i in range(len(json_list)):
            acc_list_keyword[i] = json_list[i]['scores'][0]
            label_list_keyword.append(json_list[i]['labels'][0]) 
            acc_list_raw[i] = json_raw_list[i]['scores'][0]
            label_list_raw.append(json_raw_list[i]['labels'][0]) 

        keyword_label_dic = {}
        for label in label_list_keyword:
            if label not in keyword_label_dic:
                keyword_label_dic[label] = 1
            else:
                keyword_label_dic[label] += 1

        text_label_dic = {}
        for label in label_list_raw:
            if label not in text_label_dic:
                text_label_dic[label] = 1
            else:
                text_label_dic[label] += 1

        # keyword_sorted_dic = sorted(keyword_label_dic.items(), key=lambda x:x[1], reverse = True)
        # text_sorted_dic = sorted(text_label_dic.items(), key=lambda x:x[1], reverse = True)
        
        #new label space
        label_space = read_label_space()

        for key in text_label_dic:
            if (text_label_dic[key] <= 6) and (key in label_space):
                print(key)
                label_space.remove(key)
        
        write_to_label_space(label_space)

        #delete majority label
        major_label_list = label_deletion(major_label_list, text_label_dic, keyword_label_dic)
        for label in major_label_list:
            if label in label_space:
                label_space.remove(label)
        write_to_new_label_space(label_space, iter_index)

        with open('../datasets/Amazon-531/llm_cluster_result/majority_label_list.txt', 'a') as the_file:
            the_file.write(str(major_label_list) + '\n')

        print(major_label_list)        

        zero_shot_training(iter_index, 'llama_label', 'keyword')
        zero_shot_training(iter_index, 'train_texts_split', 'text')

        final_add_label = self_training(iter_index)
        with open('../datasets/Amazon-531/llm_cluster_result/add_label_list.txt', 'a') as the_file:
            the_file.write(str(final_add_label) + '\n')

        with open('../datasets/Amazon-531/llm_cluster_result/update_labelspace.txt', 'a') as the_file:
            for label in final_add_label:
                the_file.write(label + '\n')

        print(final_add_label)

    
if __name__ == "__main__":
    main()