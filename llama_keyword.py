from transformers import AutoTokenizer
import transformers
import torch
from argparse import ArgumentParser
import os
from OpenWordMLTC.keyword_generator.get_prompt import create_prompt

def keyphrase_clearning(documents):
    documents = documents.split("Result: ")[1:]
    cleaned_documents = []
    for i, doc in enumerate(documents):
        if doc.find('[label]') != -1 and doc.find('[/label]') != -1:
            new_doc = doc.replace('[label] ', '').replace(' [/label]', '').replace('"', '')
        elif doc.find('coarse-grained labels') != -1 and doc.find('fine-grained labels') != -1:
            coarse_grain= doc.split('coarse-grained labels')[1].split('fine-grained labels')[0].split('"')
            coarse_grain_label = ''
            for i in range(1,len(coarse_grain)-1, 2):
                coarse_grain_label += coarse_grain[i] + ', '
            fine_grain= doc.split('fine-grained labels')[1].split('.')[0].split('"')
            fine_grain_label = ''
            for i in range(1,len(fine_grain)-1, 2):
                fine_grain_label += fine_grain[i] + ', '
            label = coarse_grain_label + fine_grain_label
            new_doc = doc.split(':')[0] + ': '  + label + '\n'
        elif doc.find('coarse-grained labels') != -1:
            coarse_grain= doc.split('coarse-grained labels')[1].split('.')[0].split('"')
            coarse_grain_label = ''
            for i in range(1,len(coarse_grain)-1, 2):
                coarse_grain_label += coarse_grain[i] + ', '
            new_doc = doc.split(':')[0] + ': '  + coarse_grain_label + '\n'
        elif doc.split(': ')[1][:1] == '\n':
            new_doc = doc.split(':')[0] + ': ' + '\n'
        else:
            new_doc = doc
        cleaned_documents.append(new_doc)
    return cleaned_documents


def main(args):
    file = open(f'{args.path}/{args.task}/{args.data_dir}', 'r')
    documents = file.readlines() 

    batch_size = args.batch_size

    # Hugging face repo name
    model = args.model #chat-hf (hugging face wrapper version)

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        batch_size = batch_size,
        torch_dtype=torch.float16,
        device_map= 'auto', # if you have GPU    
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    prompt = create_prompt(args.task, args.model)

    with open(f'{args.path}/{args.task}/{args.output_dir}', 'a') as the_file:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            refine_batch = []
            index_list = []
            for doc in batch:
                number_index = doc.split(" ")[0]
                replaced_text = prompt.replace('[DOCUMENTS]', doc[len(number_index):].strip())
                refine_batch.append(replaced_text)
                index_list.append(number_index)
            sequences = pipeline(
                refine_batch,
                do_sample=True,
                top_k=10,
                top_p = 0.9,
                temperature = 0.2,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            for number_index, seq in zip(index_list, sequences):
                text = seq[0]["generated_text"]
                index = text.find('and [/label].') + len('and [/label].')
                text = text[index:]
                if text.find('[label]') != -1 and text.find('[/label]') != -1:
                    #find the start and end index of the label
                    start = text.find('[label]')
                    end = text.find('/label]')
                    #extract the label
                    label = text[start:end + len('/label]')]
                    print(label)
                elif text.find('<<label>>') != -1 and text.find('/label>>') != -1:
                    start = text.find('<<label>>')
                    end = text.find('/label>>')
                    label = text[start:end + len('/label>>')]
                    print(label)
                else:
                    label = text

                the_file.write("Result: " + number_index + ": " + label + '\n')
                print("Result: " + number_index + ": " + label)

    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--output_dir", type=str, default="llama_label2_50.txt")
    args = parser.parse_args()

    main(args)