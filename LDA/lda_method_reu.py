#latent Dirichlet allocation (LDA) 

import numpy as np
import lda
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re

nltk.download('stopwords')
stopwords = stopwords.words("english")


# Function to read and parse the text file into documents
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

# Load and parse your documents
# #reuters 2999 - 8841
reuters_bertopic = 85 + 50 #remove duplicate
reuters_lines = 8841

filename = '../datasets/Reuters-21578_train_texts_50.txt'  # Change this to the path of your text file


documents = load_and_parse_documents(filename, reuters_lines)

# Preprocess documents and create a document-term matrix
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Adjust as needed
X = vectorizer.fit_transform(documents).toarray()
vocab = vectorizer.get_feature_names_out()

# Apply LDA
model = lda.LDA(n_topics=reuters_bertopic, n_iter=1500, random_state=42)
model.fit(X)

# Display the top words for each topic
topic_word = model.topic_word_
n_top_words = 8

top_1 = []
top_2 = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    top_1.append(topic_words[0])
    top_2.append(topic_words[0] + " " + topic_words[1])

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
