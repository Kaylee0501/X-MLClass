#non-negative matrix factorization (NMF)
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# A = tfidf_vectorizer.transform(texts)
# W = nmf.components_
# H = nmf.transform(A)

# A = artical x words
# W = topics x words    (topics)
# H = artical x topics  (coefficient)

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = stopwords.words("english")

#read and process based on the document id in the beginning
def load_and_parse_documents(filename, line_number):
    documents = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for _ in range(line_number):
            line = file.readline()
            # Check if you've reached the end of the file
            if not line:
                break
            paragraph = line.strip()
            if paragraph == "":
                continue
            leading_digit = re.match(r'^\d+', paragraph)
            if leading_digit:
                digit = int(leading_digit.group())
                documents.setdefault(digit, "")
                documents[digit] += paragraph[leading_digit.end():].strip() + " "
                
    return list(documents.values())

def preprocess_text(text):
    # Simple preprocessing: lowercasing and removing non-alphabetic characters
    text = text.lower()
    # text = re.sub('[^a-z\s]', '', text)  # Remove all non-alphabetical characters
    # text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    # Remove stopwords
    tokens = [word for word in text.split() if word not in stopwords]
    return ' '.join(tokens)

filename = '../datasets/Reuters-21578_train_texts_50.txt'  # Change this to the path of your text file
# #reuters 2999 - 8841
reuters_bertopic = 85  #remove duplicate
reuters_lines = 8841

texts = load_and_parse_documents(filename, reuters_lines)



tfidf_vectorizer = TfidfVectorizer(
    # min_df=10,  minimum document frequency for a term to be included in the vocabulary
    # max_df=0.85, maximum document frequency for a term to be included
    max_features=5000,
    # ngram_range=(1, 2),
    stop_words=stop_words
)

tfidf = tfidf_vectorizer.fit_transform(texts)

nmf = NMF(
    n_components=reuters_bertopic,  # Number of topics
    init='nndsvd',
    random_state=42
).fit(tfidf)

top_1 = []
top_2 = []
# Extracting topics and their top words
for i, topic in enumerate(nmf.components_):
    top_words = [tfidf_vectorizer.get_feature_names_out()[index] for index in topic.argsort()[:-10 - 1:-1]]
    print(f"Topic {i}:")
    print(" ".join(top_words))
    top_1.append(top_words[0])  # Store the top word of each topic
    if len(top_words) > 1:
        top_2.append(top_words[0] + " " + top_words[1])  # Store the top 2 words of each topic, concatenated

# # There's no need to check for uniqueness as top_1 and top_2 are constructed from ordered components and will inherently have unique entries per topic
# print("Top 1 words from each topic:")
# print(top_1)
# print("Top 2 words (pairs) from each topic:")
# print(top_2)


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