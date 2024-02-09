#all the data from alldatasets.py
from alldatasets import aapd_gt, reuters_gt, rcv1_gt, amazon_gt, dbpedia_gt
from alldatasets import aapd_labels, amazon_labels, dbpedia_labels, rcv1_labels, reuters_labels
#from alldatasets import aapd_labels_2, amazon_labels_2, dbpedia_labels_2, rcv1_labels_2, reuters_labels_2
from alldatasets import aapd_llama, amazon_llama, dbpedia_llama, rcv1_llama, reuters_llama
from alldatasets import aapd_labels_2_combined, reuters_labels_2_combined, rcv1_labels_2_combined, amazon_labels_2_combined, dbpedia_labels_2_combined

#lda
from alldatasets import aapd_lda, reuters_lda, rcv1_lda, amazon_lda, dbpedia_lda, aapd_127_lda
from alldatasets import aapd_lda_2, reuters_lda_2, rcv1_lda_2, amazon_lda_2, dbpedia_lda_2, aapd_127_lda_2


from sentence_transformers import SentenceTransformer, util


##
# encode test_word and labelset into embeddings, compute cosine similarity
# return the maximum similarity score and the label with the highest similarity score. 
def compare_word_similarity(test_word, labelset, model):
    query_embedding = model.encode(test_word)
    passage_embedding = model.encode(labelset)
    sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
    max_score = max(sim_scores)
    max_label = labelset[sim_scores.argmax()]
    return max_score, max_label

##
# For each word in gt, it finds the label with the highest similarity score above the threshold 
# adds the word-label pair to the pairs dictionary.
# remove the selected label from gt and labels to ensure they are not considered again
# until no more words in gt with similarity scores above the threshold.
# return pairs with gt-label pairs that meet the threshold
def result(gt, labels, threshold):
    #dictionary to store the gt label and label generated
    pairs = {}
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda:1')

    while gt:
        #strore the maximum score for each word
        scores = {}
        for word in gt:
            maxscore, according_label = compare_word_similarity(word, labels, model)
            scores[word] = (maxscore, according_label)
        # Find the word with the highest score
        highest_score_word = max(scores, key=lambda k: scores[k][0])
        highest_score, highest_score_label = scores[highest_score_word]


        if highest_score > threshold:
            # print(f"Word: {highest_score_word}")
            # print(f"Max Similarity Score: {highest_score}")
            # print(f"Label with Max Similarity: {highest_score_label}")
            pairs[highest_score_word] = highest_score_label
            # Remove the selected word and corresponding label from gt and labels
            gt.remove(highest_score_word)
            labels.remove(highest_score_label)
        else:
            break
        # print("=" * 50 + '\n')

    return pairs

# Make a copy to avoid modifying the original lists

total_gt_length = len(aapd_gt)

threshold = 0.6
labels = aapd_llama.copy()
gt = aapd_gt.copy()
pairs = result(gt, labels, threshold=threshold)
# Print the resulting pairs
for word, label in pairs.items():
    print(f"Groundtruth: {word}, Label: {label}")

assert len(gt) + len(pairs) == total_gt_length, "# of remainding labels in groundtruth + # pairs is not equal to the total labels in groundtruth"
print(f"There are {total_gt_length} labels in groundtruth. There are {len(pairs)} pairs in the end with a threshold of {threshold}.")
print(f"The coverage is {len(pairs)} / {total_gt_length} = {len(pairs)/total_gt_length}")

print("=" * 50)

threshold = 0.65
labels = aapd_llama.copy()
gt = aapd_gt.copy()
pairs = result(gt, labels, threshold=threshold)
# Print the resulting pairs
for word, label in pairs.items():
    print(f"Groundtruth: {word}, Label: {label}")
assert len(gt) + len(pairs) == total_gt_length, "# of remainding labels in groundtruth + # pairs is not equal to the total labels in groundtruth"
print(f"There are {total_gt_length} labels in groundtruth. There are {len(pairs)} pairs in the end with a threshold of {threshold}.")
print(f"The coverage is {len(pairs)} / {total_gt_length} = {len(pairs)/total_gt_length}")

print("=" * 50)

threshold = 0.7
labels = aapd_llama.copy()
gt = aapd_gt.copy()
pairs = result(gt, labels, threshold=threshold)
# Print the resulting pairs
for word, label in pairs.items():
    print(f"Groundtruth: {word}, Label: {label}")

assert len(gt) + len(pairs) == total_gt_length, "# of remainding labels in groundtruth + # pairs is not equal to the total labels in groundtruth"
print(f"There are {total_gt_length} labels in groundtruth. There are {len(pairs)} pairs in the end with a threshold of {threshold}.")
print(f"The coverage is {len(pairs)} / {total_gt_length} = {len(pairs)/total_gt_length}")

# def compare_word_similarity(test_word, labelset): 
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device ='cuda:1')
#     query_embedding = model.encode(test_word)
#     passage_embedding = model.encode(labelset)
#     sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
#     # max_score = max(sim_scores)
#     # max_label = labelset[sim_scores.argmax()]
#     # return sim_scores, max_score, test_word, max_label

#     # Sort the scores and get the indices of the top three scores
#     top_indices = (-sim_scores).argsort()[:3]
#     # Extract the top three scores and labels
#     top_scores = [sim_scores[i] for i in top_indices]
#     top_labels = [labelset[i] for i in top_indices]
#     return top_scores, test_word, top_labels


# #for all labels in ground truth, we pick the label with the highest similarity score out
# #delete the label in ground truth, and delete the corresponding label in labels generated
# #and then pick the next highest pair out

# #return the pairs whose score should be greater than a threshold = 0.65
# def result(labels, gt, threshold):
#     pairs = {}
#     for word in gt[:]:
#         top_scores, test_word, top_labels = compare_word_similarity(word, labels)
#         if top_scores[0] > threshold:
#             print(f"Word: {word}")
#             print(f"Max Similarity Score: {top_scores[0]}")
#             print(f"Label with Max Similarity: {top_labels[0]}")
#             pairs[word] = top_labels[0]
#             # Remove the selected word and corresponding label from gt and labels
#             gt.remove(word)
#             labels.remove(top_labels[0])
#     return pairs

# # Make a copy to avoid modifying the original list
# labels = aapd_labels.copy()
# gt = aapd_gt.copy()
# pairs = result(labels, gt, threshold = 0.65)

# # Print the resulting pairs
# for word, label in pairs.items():
#     print(f"Groundtruth: {word}, Label: {label}")