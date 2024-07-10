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


def get_tail_label(keyphrase_subset, cluster_num, task, cluster_model):
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
        docs.append([instruction, line])
    
    model = INSTRUCTOR('hkunlp/instructor-large', device='cuda')
    embeddings = model.encode(docs)
    print(embeddings)
    print(embeddings.shape)

    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    umap_model.fit(embeddings)
    umap_embeddings = np.nan_to_num(umap_model.transform(embeddings))

    if cluster_model == 'KMeans':
        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters= cluster_num)
        clustering_model.fit(umap_embeddings)
        cluster_assignment = clustering_model.labels_
        print(cluster_assignment, cluster_assignment.shape)

        select_tail_label = []
        for i in range(cluster_num):
            index = np.argsort(clustering_model.transform(umap_embeddings)[:,i])[-1]
            tail_label = keyphrase_subset[index].strip()
            select_tail_label.append(tail_label)
            if i % 50 == 0:
                print(i)
    else:
        clustering_model = GaussianMixture(n_components = cluster_num, random_state=42)
        clustering_model.fit(umap_embeddings)
        predict_label = clustering_model.predict(umap_embeddings)
        means = clustering_model.means_

        select_tail_label = []
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
            sort_index = np.argsort(dis_array)[-1]
            real_index = embedding_index[sort_index]
            tail_label = keyphrase_subset[real_index].strip()
            select_tail_label.append(tail_label)

            if label % 50 == 0:
                print(label)

    return select_tail_label




def main(args):
    cluster_num = args.cluster_size
    file = open(f'{args.path}/{args.task}/{args.keyphrase_dir}', 'r')
    documents = file.readlines()

    file1 = open(f'{args.path}/{args.task}/{args.data_dir}', 'r')
    label_class = file1.readlines()
    label_candidate = [label.strip() for label in label_class]


    keyphrase_subset = []
    for i, doc in enumerate(documents):
        if int(doc.split(": ")[0]) < args.dynamic_iter:
            keyphrase_subset.append(doc.split(": ")[1].strip())
        else:
            chunk_size = i
            break
    
    keyword_subset = []
    for labels in keyphrase_subset:
        for label in labels.split(', '):
            keyword_subset.append(label)

    print(keyword_subset[:10])
    

    select_tail_label = get_tail_label(keyword_subset, cluster_num, args.task, args.cluster_model)
    
    with open(f'{args.path}/{args.task}/{args.output_dir}', 'w') as file:
        for label in select_tail_label:
            file.write(label + '\n')

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/init_label_space.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="keyphrase_candidate/llama2_label_50.txt")
    parser.add_argument("--task", type=str, default='Amazon-531')
    parser.add_argument("--dynamic_iter", type=int, default=14000)
    parser.add_argument("--cluster_size", type=int, default=84, choices=[84, 143, 215, 422, 398])
    parser.add_argument("--cluster_model", type=str, default="GMM")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="llama2/tail_label.txt")
    args = parser.parse_args()

    main(args)