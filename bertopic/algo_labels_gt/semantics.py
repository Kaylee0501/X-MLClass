#all the data from alldatasets.py
from alldatasets import aapd_gt, reuters_gt, rcv1_gt, amazon_gt, dbpedia_gt
from alldatasets import aapd_labels, amazon_labels, dbpedia_labels, rcv1_labels, reuters_labels
from alldatasets import aapd_labels_2, amazon_labels_2, dbpedia_labels_2, rcv1_labels_2, reuters_labels_2
from alldatasets import aapd_llama, amazon_llama, dbpedia_llama, rcv1_llama, reuters_llama

from sentence_transformers import SentenceTransformer, util
def compare_word_similarity(test_word, labelset): 
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 0)
    query_embedding = model.encode(test_word)
    passage_embedding = model.encode(labelset)
    sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
    # max_score = max(sim_scores)
    # max_label = labelset[sim_scores.argmax()]
    # return sim_scores, max_score, test_word, max_label

    # Sort the scores and get the indices of the top three scores
    top_indices = (-sim_scores).argsort()[:3]
    # Extract the top three scores and labels
    top_scores = [sim_scores[i] for i in top_indices]
    top_labels = [labelset[i] for i in top_indices]
    return sim_scores, top_scores, test_word, top_labels


# write_to_path = 'labels_in_gt_aapd.txt'
# labels = aapd_labels
# gt = aapd_gt

write_to_path = 'labels_in_gt_dbpedia_2.txt'
labels = dbpedia_labels_2
gt = dbpedia_gt

# write_to_path = 'labels_in_gt_aapd_llama.txt'
# labels = aapd_llama
# gt = aapd_gt

# # Iterate through each word in ground truth
with open(write_to_path, 'w') as file:
    for word in labels:
        # sim_scores, max_score, test_word, max_label = compare_word_similarity(word, aapd_gt)
        sim_scores, top_scores, test_word, top_labels = compare_word_similarity(word, gt)
        print(word, top_scores[0], top_labels[0])
        # Print the results for each word
        # print(f"Word: {word}")
        # print("Similarity Scores:", sim_scores)
        # print("Highest Similarity Score:", max_score)
        # print("Test word in the labels:", test_word)
        # print("Corresponding Ground Truth Label:", max_label)
        # print("=" * 50 + '\n')
        # Write the results to the file
        file.write(f"Word: {word}\n")
        file.write("Similarity Scores: {}\n".format(sim_scores))
        # file.write("Highest Similarity Score: {}\n".format(max_score))
        # file.write("Corresponding Ground Truth Label: {}\n".format(max_label))
        file.write(f"top 3 options\n")
        for score, label in zip(top_scores, top_labels):
            file.write(f"Similarity Score: {score:.4f}, Label: {label}\n")
        file.write("=" * 50 + '\n')

# for word in labels:
#     sim_scores, top_scores, test_word, top_labels = compare_word_similarity(word, gt)
#     print(word, top_scores[0], top_labels[0])
    # print(word, top_scores[1], top_labels[1])
    # print(word, top_scores[2], top_labels[2])
