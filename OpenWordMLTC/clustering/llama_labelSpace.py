import os
import numpy as np
from transformers import AutoTokenizer
import transformers
import torch

file1 = open('../datasets/Reuters-21578/llm_cluster_result/doc_cluster_50_3000.txt', 'r')
documents = file1.readlines()

system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
# example_prompt = """
# I have a topic that contains the following documents:
# - the relation between pearson 's correlation coefficient and salton 's cosine measure is revealed based on the different possible values of the division of the l1 norm and the l2 norm of a vector 
# - these different values yield a sheaf of increasingly straight lines which form together a cloud of points , being the investigated relation the theoretical results are tested against the author co citation relations 
# - among 24 informetricians for whom two matrices can be constructed , based on co citations the asymmetric occurrence matrix and the symmetric co citation matrix both examples completely confirm the theoretical results 
# - the results enable us to specify an algorithm which provides a threshold value for the cosine above which none of the corresponding pearson correlations would be negative using this threshold value can be expected to optimize the visualization of the vector space"

# Based on the information about the topic above, please find one label for this topic above. Make sure you only return the output without anything else in one line.

# [/INST] Information Retrieval
# """

example_prompt = """
I have a topic that contains the following documents:
- OHIO MATTRESS &lt;OMT> MAY HAVE LOWER 1ST QTR NET Ohio Mattress Co said its first quarter, ending February 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the first quarter of fiscal
- 1986. The company said any decline would be due to expenses related to the acquisitions in the middle of the current quarter of seven licensees of Sealy Inc, as well as 82 pct of
- the outstanding capital stock of Sealy. Because of these acquisitions, it said, first quarter sales will be substantially higher than last year's 67.1 mln dlrs. Noting that it typically reports first quarter results in
- late march, said the report is likely to be issued in early April this year. It said the delay is due to administrative considerations, including conducting appraisals, in connection with the acquisitions.
  
Based on the information about the topic above, please find one label for this topic above. Please output your answer use the following format in one line:

[/INST] acquisitions
"""

# example_prompt = """
# I have a topic that contains the following documents:
# - omron hem 790it automatic blood pressure monitor with advanced omron health management software so far this machine has worked well
# - and is very simple to use . it is nice to have immediate feedback on the bloodpressure effects of my various exercises , 
# - food consumption , and relaxation or stress levels .

# Based on the information about the topic above, please find one label for this topic above. Please output your answer use the following format in one line:

# [/INST] health_personal_care
# """

main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

Based on the information about the topic above, please find one label for this topic above. Make sure you only return the output without anything else in one line.
[/INST]
"""

prompt = system_prompt + example_prompt + main_prompt

# Hugging face repo name
model = "meta-llama/Llama-2-13b-chat-hf" #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map={"" : 3} # if you have GPU
)

with open('../datasets/Reuters-21578/llm_cluster_result/50chunk_labelspace.txt', 'a') as the_file:
    for i, doc in enumerate(documents):
        replaced_text = prompt.replace('[DOCUMENTS]', doc.strip())
        sequences = pipeline(
            replaced_text,
            do_sample=True,
            top_k=10,
            top_p = 0.9,
            temperature = 0.2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        label_text = sequences[0]["generated_text"][1250:]
        index = label_text.find('[/INST]') + len('[/INST] ')
        if label_text[index] == '\n':
            index += 1
        the_file.write(str(i) + ": " + label_text[index:] + '\n')
        print(str(i) + ": " + label_text[index:])