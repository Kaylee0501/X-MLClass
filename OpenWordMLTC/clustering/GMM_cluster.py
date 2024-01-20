import os
import sklearn.cluster
import numpy as np
from sklearn.mixture import GaussianMixture
from InstructorEmbedding import INSTRUCTOR
from numpy import savetxt
import pandas as pd
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ['OPENBLAS_NUM_THREADS'] = '1'


cluster_num = 422

file1 = open('../datasets/DBPedia-298/llama_label_50.txt', 'r')
documents = file1.readlines()[:26785]
print(len(documents))

file2 = open('../datasets/DBPedia-298/train_texts_split_50.txt', 'r')
original_docs = file2.readlines()[:26785]

docs = []
instruction = "Represent documents collected from DBPedia news for clustering: "
for line in documents:
    new_line = line.split(": ")[1].strip()
    docs.append([instruction, new_line])


model = INSTRUCTOR('hkunlp/instructor-large')
embeddings = model.encode(docs)
print(embeddings)
print(embeddings.shape)

# clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters= cluster_num)
# clustering_model.fit(embeddings)
# cluster_assignment = clustering_model.labels_
# print(cluster_assignment, cluster_assignment.shape)

# data = {'Document': original_docs, 
#         'Label': cluster_assignment} 
# df = pd.DataFrame(data) 

# df.to_csv("../datasets/Reuters-21578/labelSpace/GMM_doc_label_50_test.csv", index=False)

# # with open('../datasets/AAPD/select_label_index.txt', 'a') as index_file:
# #     for content in cluster_assignment:
# #         index_file.write(str(content) + '\n')

# cluster_dic = defaultdict(list)
# for i, label in enumerate(cluster_assignment):
#         cluster_dic[label].append(i)

# select_docs_per_label = []
# for i in range(cluster_num):
#     indexs = np.argsort(clustering_model.transform(embeddings)[:,i])[:3]
#     doc_list = ''
#     for ind in indexs:
#         doc_list += original_docs[ind].strip() + ' '
#     select_docs_per_label.append(doc_list)
#     if i % 50 == 0:
#         print(i)

from umap import UMAP

umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
import numpy as np
umap_model.fit(embeddings)
umap_embeddings = np.nan_to_num(umap_model.transform(embeddings))

clustering_model = GaussianMixture(n_components = cluster_num, random_state=0)
clustering_model.fit(umap_embeddings)
predict_label = clustering_model.predict(umap_embeddings)
means = clustering_model.means_

# data = {'Document': original_docs, 
#         'Label': predict_label} 
# df = pd.DataFrame(data) 

# df.to_csv("../datasets/Reuters-21578/labelSpace/GMM_doc_label_50_test.csv", index=False)

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

with open('../datasets/DBPedia-298/llm_cluster_result/doc_cluster_50_8000.txt', 'a') as the_file:
     for content in select_docs_per_label:
        the_file.write(content + '\n')
