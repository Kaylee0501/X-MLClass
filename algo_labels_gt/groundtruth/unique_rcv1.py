file_path = "./RCV1_train_labels.txt"

unique_words = set()
total_count = 0
with open(file_path, 'r') as file:
    for line in file:
        #Remove leading and trailing whitespace from the line, skip empty lines
        line = line.strip()
        if not line: continue
        # Split each line into words
        # get each phrase from splitting by ;
        words = line.split(';')
        for word in words:
            #only add uniques words, and remove leading and trailing whitespace from each phrase 
            word = word.strip()
            #skip empty ones
            if not word:
                continue
            total_count += 1
            unique_words.add(word)

word_mapping = {
    "ec": "european community"
}

unique_words_list = list(unique_words)

for old_word, new_word in word_mapping.items():
    for i, word in enumerate(unique_words_list):
        if old_word in word:
            unique_words_list[i] = word.replace(old_word, new_word)

print("unique", unique_words_list)
print("total count is", total_count)
print("unique count is", len(unique_words_list))