from transformers import AutoTokenizer
import transformers
import sklearn.cluster
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from InstructorEmbedding import INSTRUCTOR
from argparse import ArgumentParser
from umap import UMAP
import os
from get_prompt import create_prompt, create_final_prompt


def get_top_chunks(keyphrase_subset, original_docs, cluster_num, task):
    docs = []
    if task == 'AAPD':
        instruction = "Represent documents collected from computer science paper abstract for clustering: "
    elif task == 'Amazon-531':
        instruction = "Represent documents collected from Amazon review data for clustering: "
    elif task == 'DBPedia-298':
        instruction = "Represent documents collected from Wikipedia facts for clustering: "
    elif task == 'Reuters-21578':
        instruction = "Represent documents collected from Reuters News Wire for clustering: "
    elif task == 'RCV1':
        instruction = "Represent documents collected from the Reuters newswire for clustering: "
    else:
        raise NotImplementedError("Task not implemented")
    
    for line in keyphrase_subset:
        new_line = line.split(": ")[1].strip()
        docs.append([instruction, new_line])
    
    model = INSTRUCTOR('hkunlp/instructor-large', device='cuda')
    embeddings = model.encode(docs)
    print(embeddings)
    print(embeddings.shape)

    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    umap_model.fit(embeddings)
    umap_embeddings = np.nan_to_num(umap_model.transform(embeddings))

    if task == 'Amazon-531':
        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters= cluster_num)
        clustering_model.fit(umap_embeddings)
        cluster_assignment = clustering_model.labels_
        print(cluster_assignment, cluster_assignment.shape)

        select_docs_per_label = []
        for i in range(cluster_num):
            indexs = np.argsort(clustering_model.transform(umap_embeddings)[:,i])[:3]
            doc_list = ''
            for ind in indexs:
                doc_list += original_docs[ind].strip() + ' '
            select_docs_per_label.append(doc_list)
            if i % 50 == 0:
                print(i)
    else:
        clustering_model = GaussianMixture(n_components = cluster_num, random_state=42)
        clustering_model.fit(umap_embeddings)
        predict_label = clustering_model.predict(umap_embeddings)
        means = clustering_model.means_

        select_docs_per_label = []
        for label in range(cluster_num):
            embedding_list = []
            embedding_index = []
            cur_mean = means[label]
            for i, predict in enumerate(predict_label):
                if predict == label:
                    embedding_list.append(umap_embeddings[i])
                    embedding_index.append(i)
            dis_array = np.zeros(len(embedding_list))
            for index, embed in enumerate(embedding_list):
                distance = np.linalg.norm(embed - cur_mean)
                dis_array[index] = distance
            sort_indexs = np.argsort(dis_array)[:3]
            doc_list = ''
            for cur_index in sort_indexs:
                real_index = embedding_index[cur_index]
                doc_list += original_docs[real_index].strip() + ' '
            select_docs_per_label.append(doc_list)

            if label % 50 == 0:
                print(label)

    return select_docs_per_label

def label_cleaning(label):
    if label.find('[label]') != -1 and label.find('[/label]') != -1:
        coarse_grain = label.split('"')
        coarse_grain_label = ''
        for i in range(1,len(coarse_grain)-1, 2):
            coarse_grain_label += coarse_grain[i] + '; '
        new_doc = coarse_grain_label + '\n'
    elif label.find('coarse-grained ') != -1 and label.find('fine-grained ') != -1:
        coarse_grain= label.split('coarse-grained ')[1].split('fine-grained ')[0].split('"')
        coarse_grain_label = ''
        for i in range(1,len(coarse_grain)-1, 2):
            coarse_grain_label += coarse_grain[i] + '; '
        fine_grain= label.split('fine-grained ')[1].split('.')[0].split('"')
        fine_grain_label = ''
        for i in range(1,len(fine_grain)-1, 2):
            fine_grain_label += fine_grain[i] + '; '
        label = coarse_grain_label + fine_grain_label
        new_doc = label + '\n'
    elif label.find('the label for ') != -1:
        print("PASS")
        coarse_grain= label.split('the label for ')[1].split('.')[0].split('"')
        coarse_grain_label = ''
        for i in range(1,len(coarse_grain)-1, 2):
            coarse_grain_label += coarse_grain[i] + '; '
        new_doc = coarse_grain_label + '\n'
    else:
        new_doc = label
    return new_doc

def gen_init_labelspace(batch_size, documents, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        batch_size = batch_size,
        torch_dtype=torch.float16,
        device_map= 'auto', # if you have GPU    
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    prompt = create_final_prompt(args.task, args.model)

    with open(f'{args.path}/{args.task}/{args.output_dir}/{args.output_file}', 'a') as the_file:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            refine_batch = []
            index_list = []
            for doc in batch:
                number_index = doc.split(" ")[0]
                replaced_text = prompt.replace('[DOCUMENTS]', doc[len(number_index):].strip())
                refine_batch.append(replaced_text)
                index_list.append(number_index)
            sequences = pipeline(
                refine_batch,
                do_sample=True,
                top_k=10,
                top_p = 0.9,
                temperature = 0.2,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            for number_index, seq in zip(index_list, sequences):
                text = seq[0]["generated_text"]
                index = text.find('and [/label].') + len('and [/label].')
                text = text[index:]
                if text.find('[label]') != -1 and text.find('[/label]') != -1:
                    #find the start and end index of the label
                    start = text.find('[label]')
                    end = text.find('/label]')
                    #extract the label
                    label = text[start:end + len('/label]')]
                    print(label)
                elif text.find('<<label>>') != -1 and text.find('/label>>') != -1:
                    start = text.find('<<label>>')
                    end = text.find('/label>>')
                    label = text[start:end + len('/label>>')]
                    print(label)
                else:
                    label = text

                the_file.write(label_cleaning(label))
                print("Result: " + number_index + ": " + label)



def main(args):
    cluster_num = args.cluster_size
    file = open(f'{args.path}/{args.task}/{args.keyphrase_dir}', 'r')
    documents = file.readlines()

    keyphrase_subset = []
    for i, doc in enumerate(documents):
        if int(doc.split(": ")[0]) < args.dynamic_iter:
            keyphrase_subset.append(doc)
        else:
            chunk_size = i
            break
    
    file2 = open(f'{args.path}/{args.task}/{args.data_dir}', 'r')
    original_docs = file2.readlines()[:chunk_size]
    print(len(original_docs))

    folder_path = f'{args.path}/{args.task}/{args.output_dir}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    select_docs_per_label = get_top_chunks(keyphrase_subset, original_docs, cluster_num, args.task)
    gen_init_labelspace(args.batch_size, select_docs_per_label, args)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="llama2_label_50.txt")
    parser.add_argument("--task", type=str, default='Amazon-531')
    parser.add_argument("--dynamic_iter", type=int, default=14000)
    parser.add_argument("--cluster_size", type=int, default=398, choices=[84, 143, 215, 422, 398])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="llama2")
    parser.add_argument("--output_file", type=str, default="init_labelspace.txt")
    args = parser.parse_args()

    main(args)