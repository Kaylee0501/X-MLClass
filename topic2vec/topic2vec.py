from top2vec import Top2Vec

#aapd 2999 - 11033
aapd_bertopic = 80  #remove duplicate
aapd_lines = 11033
fileaapd = open('../datasets/AAPD_train_texts_50.txt', 'r')
new_documents = fileaapd.readlines()[:aapd_lines]

# #reuters 2999 - 8841
# reuters_bertopic = 85 #remove duplicate
# reuters_lines = 8841
# fileaapd = open('../datasets/Reuters-21578_train_texts_50.txt', 'r')
# new_documents = fileaapd.readlines()[:reuters_lines]


# #rcv1 2999 - 14125
# rcv1_bertopic = 90  #remove duplicate
# rcv1_lines = 14125
# fileaapd = open('../datasets/RCV1_train_texts_50.txt', 'r')
# new_documents = fileaapd.readlines()[:rcv1_lines]

# #amazon 13999 - 31918
# amazon_bertopic = 256  #remove duplicate
# amazon_lines = 31918
# fileaapd = open('../datasets/Amazon-531_train_texts_50.txt', 'r')
# new_documents = fileaapd.readlines()[:amazon_lines]

# #dbpedia 7999 - 26782
# dbpedia_bertopic =  161  #remove duplicate
# dbpedia_lines = 26782
# fileaapd = open('../datasets/DBPedia-298_train_texts_50.txt', 'r')
# new_documents = fileaapd.readlines()[:dbpedia_lines]


print(len(new_documents), type(new_documents))



embedding_model = "all-MiniLM-L6-v2"
umap_args = {'n_neighbors': 5,
             'n_components': 5,
             'min_dist': 0.0,
             'metric': 'cosine',
             "random_state": 42}
hdbscan_args = {'min_cluster_size': 20,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'prediction_data': 'True'}

model = Top2Vec(documents=new_documents, 
    embedding_model=embedding_model, 
    speed="learn", 
    workers=8,
    umap_args = umap_args, 
    hdbscan_args = hdbscan_args)



print("number of topics in total in the begining:", model.get_num_topics() )
topic_words, word_scores, topic_nums = model.get_topics()
# for topic in topic_nums:
#     model.generate_topic_wordcloud(topic)
print(topic_words)
top_1 = []
top_2 = []
for inner_list in topic_words:
    top_1.append(inner_list[0])
    top_2.append(inner_list[0] + " " + inner_list[1])
# Check if lengths are equal and print messages
if len(top_1) == len(set(top_1)):
    print("For top 1, the lengths are equal, it is " + str(len(top_1)))
else:
    print("For top 1, the lengths are not equal, before is "+ str(len(top_1))+", after is "+ str(len(set(top_1))))
if len(top_2) == len(set(top_2)):
    print("For top 2, the lengths are equal, it is " + str(len(top_2)))
else:
    print("For top 2, the lengths are not equal, before is "+ str(len(top_2))+", after is "+ str(len(set(top_2))))
#print out the list without duplicsated topics
print("top1 list is")
print(list(set(top_1)))
print("top2 list is" )
print(list(set(top_2)))




#reduced discarded, since there are redundant labels in the list
# print("=" * 50)

# # print("number of topics in total after the reduction:")
# # #Reduce the number of topics discovered by Top2Vec.
# # model.hierarchical_topic_reduction(num_topics=aapd_bertopic)
# # print(model.get_num_topics(reduced = True))
# # #top 50 words are returned in order of semantic similarity to topic with scores and unique number of every topic will be returned 
# # topic_words, word_scores, topic_nums = model.get_topics(3)
# # print(topic_words)
# # sizes = [len(inner_list) for inner_list in topic_words]
# print("number of topics in total after the reduction:")
# model.hierarchical_topic_reduction(num_topics=aapd_bertopic)
# print(model.get_num_topics(reduced = True))
# topic_words, word_scores, topic_nums = model.get_topics(reduced = True)
# # for topic in topic_nums:
# #     model.generate_topic_wordcloud(topic)
# print(topic_words)
# top_1 = []
# top_2 = []
# for inner_list in topic_words:
#     top_1.append(inner_list[0])
#     top_2.append(inner_list[0] + " " + inner_list[1])
# # Check if lengths are equal and print messages
# if len(top_1) == len(set(top_1)):
#     print("For top 1, the lengths are equal, it is " + str(len(top_1)))
# else:
#     print("For top 1, the lengths are not equal, before is "+ str(len(top_1))+", after is "+ str(len(set(top_1))))
# if len(top_2) == len(set(top_2)):
#     print("For top 2, the lengths are equal, it is " + str(len(top_2)))
# else:
#     print("For top 2, the lengths are not equal, before is "+ str(len(top_2))+", after is "+ str(len(set(top_2))))
# print("top1 list is")
# print(top_1)
# print("top2 list is" )
# print(top_2)

