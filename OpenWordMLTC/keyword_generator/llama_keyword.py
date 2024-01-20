from transformers import AutoTokenizer
import transformers
import torch

# file1 = open('../datasets/Reuters-21578/train_raw_texts.txt', 'r')
# raw_label_set = file1.readlines()
# chuck = 65
# with open('../datasets/Reuters-21578/train_texts_split.txt', 'a') as the_file:
#     for i, row in enumerate(raw_label_set):
#         new_row = " ".join(row.split())
#         row_list = new_row.split()
#         while len(row_list) >= chuck:
#             new_row = row_list[:chuck]
#             document = " ".join(new_row)
#             the_file.write(f'{i} {document}\n')
#             row_list = row_list[chuck:]
#         if len(row_list) > 4:
#             document = " ".join(row_list)
#             the_file.write(f'{i} {document}\n')

file1 = open('../datasets/RCV1-V2/test_texts_split_50.txt', 'r')
documents = file1.readlines()       

# Hugging face repo name
model = "meta-llama/Llama-2-13b-chat-hf" #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map= {'':0} # if you have GPU
)

system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for multi-label text classification.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
example_prompt = """
I have a topic that contains the following documents:
- divid minist minist wary elect elect opinion sunday sunday poll rule rule rule blair french extent convinc polit left left shadow friday won consent point attach
- bring singl singl singl singl singl singl singl vow independ line hold time busi decid decid lose early labor labor labor labor labor labor polic pow year european european 
- european european european tony involut gener gener john cost mat newspap econom join join join join adopt month britain britain britain britain britain wav due countr lead lead brown brown 
- brown ahead ahead ahead conserv conserv conserv consid consid vote prim gordon deep dang major succeed appar told told recogn pledg pledg sovereignt nation secur cond monet concern parlia pro wrangl 
- interest agree call party party party party stress prev substant brit union union union clear financ remain currenc currenc currenc currenc currenc currenc currenc oppos referendum referendum referendum referendum referendum

Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:

[/INST] Government/social, Economics, Domestic Policy, European Community 
"""

main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:
[/INST]
"""

prompt = system_prompt + example_prompt + main_prompt        

with open('../datasets/RCV1-V2/llama_label_test_50.txt', 'a') as the_file:
    for i, doc in enumerate(documents):
        number_index = doc.split(" ")[0]
        replaced_text = prompt.replace('[DOCUMENTS]', doc[len(number_index):].strip())
        sequences = pipeline(
            replaced_text,
            do_sample=True,
            top_k=10,
            top_p = 0.9,
            temperature = 0.2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        label_text = sequences[0]["generated_text"][1410:]
        index = label_text.find('[/INST]') + len('[/INST] ')
        if label_text[index] == '\n':
            index += 1
        the_file.write(number_index + ": " + label_text[index:] + '\n')
        print(number_index + ": " + label_text[index:])


file2 = open('../datasets/RCV1-V2/test_texts_split_250.txt', 'r')
documents2 = file2.readlines()  

with open('../datasets/RCV1-V2/llama_label_test_250.txt', 'a') as the_file:
    for i, doc in enumerate(documents2):
        number_index = doc.split(" ")[0]
        replaced_text = prompt.replace('[DOCUMENTS]', doc[len(number_index):].strip())
        sequences = pipeline(
            replaced_text,
            do_sample=True,
            top_k=10,
            top_p = 0.9,
            temperature = 0.2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        label_text = sequences[0]["generated_text"][1410:]
        index = label_text.find('[/INST]') + len('[/INST] ')
        if label_text[index] == '\n':
            index += 1
        the_file.write(number_index + ": " + label_text[index:] + '\n')
        print(number_index + ": " + label_text[index:])