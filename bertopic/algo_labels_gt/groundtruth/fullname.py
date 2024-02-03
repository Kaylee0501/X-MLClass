#for aapd
#regular expression to extract text between the first and second semicolons on each line
import re

file_path = "./AAPD_train_labels.txt"
extracted_text_list = []
remove_duplicates_list = []
with open(file_path, 'r') as file:
    for line in file:
        extracted_text = re.search(r';\s*([^;]+)\s*;', line)
        if extracted_text:
            text = extracted_text.group(1)
            if(text not in extracted_text_list ):
                remove_duplicates_list.append(text)
            extracted_text_list.append(text)
            

print("all", extracted_text_list)
print( "count is", len(extracted_text_list))

print("remove duplicates", remove_duplicates_list)
print("count is", len(remove_duplicates_list))