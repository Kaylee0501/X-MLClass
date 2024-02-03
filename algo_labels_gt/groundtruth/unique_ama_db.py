# file_path = "./groundtruth/Amazon_train_labels.txt"
file_path = "./DBPedia_train_labels.txt"

unique_words = set()
total_count = 0
with open(file_path, 'r') as file:
    for line in file:
        #skip empty lines
        if not line: continue
        #Split the line by space and skip the first element (line number)
        words = line.split()[1:]
        for word in words:
            #only add uniques words, and remove leading and trailing whitespace from each phrase 
            word = word.strip()
            #skip empty ones
            if not word:  continue
            total_count += 1
            unique_words.add(word)


unique_words_list = list(unique_words)

print("unique", unique_words_list)
print("total count is", total_count)
print("unique count is", len(unique_words_list))