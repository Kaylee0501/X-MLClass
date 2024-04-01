
import re
nums = []
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
                
                nums.append(digit)
                documents.setdefault(digit, "")
                documents[digit] += paragraph[leading_digit.end():].strip() + " "
    print(set(nums))
    return list(documents.values())

# Load and parse your documents
# #reuters 2999 - 8841
reuters_bertopic = 85 + 50 #remove duplicate
reuters_lines = 8841

filename = '../datasets/Reuters-21578_train_texts_50.txt'  # Change this to the path of your text file


documents = load_and_parse_documents(filename, reuters_lines)

print("expected num of files: 3000")
print("actual", len(set(nums)))
for i in range(1, 3000+1):
    if i not in nums:
        print (i,  "is missing.")