import fire

def create_prompt(name, model):
    if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
        system_prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant for multi-label text classification<|eot_id|><|start_header_id|>user<|end_header_id|>
        """
    elif model == 'meta-llama/Llama-2-13b-chat-hf':
        system_prompt = """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for labeling topics.
        <</SYS>>
        """
    else:
        raise NotImplementedError(f"Model {model} not found in the model list")
    
    example_prompt = get_example_prompt(name, model)
    main_prompt = get_main_prompt(name, model)

    prompt = system_prompt + example_prompt + main_prompt 

    return prompt

def create_final_prompt(name, model):
    if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
        system_prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant for multi-label text classification<|eot_id|><|start_header_id|>user<|end_header_id|>
        """
    elif model == 'meta-llama/Llama-2-13b-chat-hf':
        system_prompt = """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for labeling topics.
        <</SYS>>
        """
    else:
        raise NotImplementedError(f"Model {model} not found in the model list")
    
    example_prompt = get_final_example_prompt(name, model)
    main_prompt = get_final_main_prompt(name, model)

    prompt = system_prompt + example_prompt + main_prompt 

    return prompt


def get_example_prompt(name, model):
    if name == "AAPD":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is an abstract from computer science academic paper. 
            "the present work studies quantum and classical correlations in three qubits and four qubits general bell states , produced by operating with braid operators on the computational
            basis of states the analogies between the general three qubits and four qubits bell states and that of two qubits bell states are discussed the general bell states are shown to
            be maximal entangled , i e , with quantum correlations which are lost by tracing these states over one qubit , remaining only with classical correlations , as shown by hs decomposition" 
            Based on the information about the abstract above, the coarse-grained keyphrases are formatted as "Quantum Physics" and "Information Theory".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "Quantum Physics", "Information Theory" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>

            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
        # Example prompt demonstrating the output we are looking for
            example_prompt = """
            The following is an abstract from computer science academic paper. 
            "the present work studies quantum and classical correlations in three qubits and four qubits general bell states , produced by operating with braid operators on the computational
            basis of states the analogies between the general three qubits and four qubits bell states and that of two qubits bell states are discussed the general bell states are shown to
            be maximal entangled , i e , with quantum correlations which are lost by tracing these states over one qubit , remaining only with classical correlations , as shown by hs decomposition" 
            Based on the information about the abstract above, the coarse-grained keyphrases are formatted as "Quantum Physics" and "Information Theory".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "Quantum Physics", "Information Theory" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")

    elif name == "Amazon-531":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a customer review of a product bought from Amazon. 
            "omron hem 790it automatic blood pressure monitor with advanced omron health management software so far this machine has worked well and is very simple to use. it 
            is nice to have immediate feedback on the blood-pressure effects of my various exercises, food consumption, and relaxation or stress levels" 
            Based on the information about the topic above, the coarse-grained keyphrases are formatted as "health_personal_care"  and the fine-grained keyphrases are formatted as  "medical_supplies_equipment" and "health_monitors".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "health_personal_care", "medical_supplies_equipment", "health_monitors" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>

            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a customer review of a product bought from Amazon. 
            "omron hem 790it automatic blood pressure monitor with advanced omron health management software so far this machine has worked well and is very simple to use. it 
            is nice to have immediate feedback on the blood-pressure effects of my various exercises, food consumption, and relaxation or stress levels" 
            Based on the information about the topic above, the coarse-grained keyphrases are formatted as "health_personal_care"  and the fine-grained keyphrases are formatted as  "medical_supplies_equipment" and "health_monitors".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "health_personal_care", "medical_supplies_equipment", "health_monitors" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")

    elif name == 'DBPedia-298':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a fact from the English edition of Wikipedia. 
            "enyalioides binzayedi is a species of lizard in the genus enyalioides known from only one location in the cordillera azul national park in peru .
            the lizard is named after mohammed bin zayed al nahyan , who sponsored the field survey that led to the discovery of the species ."
            Based on the information about the abstract above, the coarse-grained keyphrases are formatted as "species" and "animal" and the fine-grained keyphrases are formatted as "reptile".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "species", "animal", "reptile" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a fact from the English edition of Wikipedia. 
            "enyalioides binzayedi is a species of lizard in the genus enyalioides known from only one location in the cordillera azul national park in peru .
            the lizard is named after mohammed bin zayed al nahyan , who sponsored the field survey that led to the discovery of the species ."
            Based on the information about the abstract above, the coarse-grained keyphrases are formatted as "species" and "animal" and the fine-grained keyphrases are formatted as "reptile".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "species", "animal", "reptile" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")

    elif name ==  'RCV1':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a newswire acticle from the Reuters newswire service. 
            "Another liability challenge to the tobacco industry headed toward a decision in court Tuesday.A jury in Marion County Superior Court was expected to begin deliberations in the case on Wednesday or Thursday. 
            Kansas Attorney General Carla Stovall discuss their states' litigation. Also Tuesday, New York City's public advocate urged Mayor Rudolph Giuliani to sue the tobacco industry to recoup health care costs of smokers. 
            The Rogers case, similar to hundreds that have been filed across the country, comes on the heels of a one earlier this month in Jacksonville, Florida, where a jury awarded $750,000 to a man who smoked for 44 years before he was stricken with lung cancer.
            Corp. is a unit of Britain's B.A.T Industries Plc.In the Indianapolis case, which was tried once before and wound up in a hung jury, the Rogers family contended that the industry peddled an addictive product that was the cause of his lung cancer.Rogers originally filed the suit himself".
            Based on the information about the topic above, the coarse-grained labels keyphrases are formatted as "corporate/industrial" and "government/social" and the fine-grained keyphrases are formatted as "legal/judicial" and "crime, law enforcement".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "corporate/industrial", "government/social", "legal/judicial", "crime, law enforcement" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a newswire acticle from the Reuters newswire service. 
            "Another liability challenge to the tobacco industry headed toward a decision in court Tuesday.A jury in Marion County Superior Court was expected to begin deliberations in the case on Wednesday or Thursday. 
            Kansas Attorney General Carla Stovall discuss their states' litigation. Also Tuesday, New York City's public advocate urged Mayor Rudolph Giuliani to sue the tobacco industry to recoup health care costs of smokers. 
            The Rogers case, similar to hundreds that have been filed across the country, comes on the heels of a one earlier this month in Jacksonville, Florida, where a jury awarded $750,000 to a man who smoked for 44 years before he was stricken with lung cancer.
            Corp. is a unit of Britain's B.A.T Industries Plc.In the Indianapolis case, which was tried once before and wound up in a hung jury, the Rogers family contended that the industry peddled an addictive product that was the cause of his lung cancer.Rogers originally filed the suit himself".
            Based on the information about the topic above, the coarse-grained keyphrases are formatted as "corporate/industrial" and "government/social" and the fine-grained keyphrases are formatted as "legal/judicial" and "crime, law enforcement".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "corporate/industrial", "government/social", "legal/judicial", "crime, law enforcement" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
        
    elif name == 'Reuters-21578':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a news acticle from the Reuters financial newswire service. 
            "ending February 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the first quarter of fiscal 1986. The company said any decline would be due to expenses related to the acquisitions 
            in the middle of the current quarter of seven licensees of Sealy Inc. Because of these acquisitions, it said, first quarter sales will be substantially higher than last year's 67.1 mln dlrs." 
            Based on the information about the topic above, the coarse-grained labels keyphrases are formatted as "earn" and "acquisitions".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "earn", "acquisitions" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a news acticle from the Reuters financial newswire service. 
            "ending February 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the first quarter of fiscal 1986. The company said any decline would be due to expenses related to the acquisitions 
            in the middle of the current quarter of seven licensees of Sealy Inc. Because of these acquisitions, it said, first quarter sales will be substantially higher than last year's 67.1 mln dlrs." 
            Based on the information about the topic above, the coarse-grained lkeyphrases are formatted as "earn" and "acquisitions".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "earn", "acquisitions" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    else:
        raise NotImplementedError(f"Name {name} not found in the dataset list")

    return example_prompt

def get_main_prompt(name, model):
    if name == "AAPD":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more subjects that best describe the paper abstract. Please find at least three coarse-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [label] and [/label].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more subjects that best describe the paper abstract. Please find at least three coarse-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    elif name == "Amazon-531":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the reviewed product. Please find two coarse-grained and two fine-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [label] and [/label].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the reviewed product. Please find two coarse-grained and two fine-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
        
    elif name == 'DBPedia-298':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more categories that best describe the Wikipedia fact. Please find two coarse-grained and two fine-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more categories that best describe the Wikipedia fact. Please find two coarse-grained and two fine-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] four labels [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    elif name == 'RCV1':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the newswire acticle. Please find two coarse-grained and two fine-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [label] and [/label].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the newswire acticle. Please find two coarse-grained and two fine-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    elif name == 'Reuters-21578':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the news acticle. Please find find at least three coarse-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the news acticle. Please find find at least three coarse-grained labels for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    else:
        raise NotImplementedError(f"Name {name} not found in the dataset list")
    
    return main_prompt

def get_final_example_prompt(name, model):
    if name == "AAPD":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is an abstract from computer science academic paper. 
            "the present work studies quantum and classical correlations in three qubits and four qubits general bell states , produced by operating with braid operators on the computational
            basis of states the analogies between the general three qubits and four qubits bell states and that of two qubits bell states are discussed the general bell states are shown to
            be maximal entangled , i e , with quantum correlations which are lost by tracing these states over one qubit , remaining only with classical correlations , as shown by hs decomposition" 
            Based on the information about the abstract above, the label for the document is "Quantum Physics".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "Quantum Physics" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
        # Example prompt demonstrating the output we are looking for
            example_prompt = """
            The following is an abstract from computer science academic paper. 
            "the present work studies quantum and classical correlations in three qubits and four qubits general bell states , produced by operating with braid operators on the computational
            basis of states the analogies between the general three qubits and four qubits bell states and that of two qubits bell states are discussed the general bell states are shown to
            be maximal entangled , i e , with quantum correlations which are lost by tracing these states over one qubit , remaining only with classical correlations , as shown by hs decomposition" 
            Based on the information about the abstract above, the label for the document is "Quantum Physics".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "Quantum Physics" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")

    elif name == "Amazon-531":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a customer review of a product bought from Amazon. 
            "omron hem 790it automatic blood pressure monitor with advanced omron health management software so far this machine has worked well and is very simple to use. it 
            is nice to have immediate feedback on the blood-pressure effects of my various exercises, food consumption, and relaxation or stress levels" 
            Based on the information about the topic above, the label for the document is "health_personal_care".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "health_personal_care" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a customer review of a product bought from Amazon. 
            "omron hem 790it automatic blood pressure monitor with advanced omron health management software so far this machine has worked well and is very simple to use. it 
            is nice to have immediate feedback on the blood-pressure effects of my various exercises, food consumption, and relaxation or stress levels" 
            Based on the information about the topic above, the label for the document is "health_personal_care".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "health_personal_care" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")

    elif name == 'DBPedia-298':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a fact from the English edition of Wikipedia. 
            "enyalioides binzayedi is a species of lizard in the genus enyalioides known from only one location in the cordillera azul national park in peru .
            the lizard is named after mohammed bin zayed al nahyan , who sponsored the field survey that led to the discovery of the species ."
            Based on the information about the abstract above, the label for the document is "species".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "species" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a fact from the English edition of Wikipedia. 
            "enyalioides binzayedi is a species of lizard in the genus enyalioides known from only one location in the cordillera azul national park in peru .
            the lizard is named after mohammed bin zayed al nahyan , who sponsored the field survey that led to the discovery of the species ."
            Based on the information about the abstract above, the label for the document is "species".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "species" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")

    elif name ==  'RCV1':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a newswire acticle from the Reuters newswire service. 
            "Another liability challenge to the tobacco industry headed toward a decision in court Tuesday.A jury in Marion County Superior Court was expected to begin deliberations in the case on Wednesday or Thursday. 
            Kansas Attorney General Carla Stovall discuss their states' litigation. Also Tuesday, New York City's public advocate urged Mayor Rudolph Giuliani to sue the tobacco industry to recoup health care costs of smokers. 
            The Rogers case, similar to hundreds that have been filed across the country, comes on the heels of a one earlier this month in Jacksonville, Florida, where a jury awarded $750,000 to a man who smoked for 44 years before he was stricken with lung cancer.
            Corp. is a unit of Britain's B.A.T Industries Plc.In the Indianapolis case, which was tried once before and wound up in a hung jury, the Rogers family contended that the industry peddled an addictive product that was the cause of his lung cancer.Rogers originally filed the suit himself".
            Based on the information about the topic above, the label for the document is "corporate/industrial".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "corporate/industrial" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a newswire acticle from the Reuters newswire service. 
            "Another liability challenge to the tobacco industry headed toward a decision in court Tuesday.A jury in Marion County Superior Court was expected to begin deliberations in the case on Wednesday or Thursday. 
            Kansas Attorney General Carla Stovall discuss their states' litigation. Also Tuesday, New York City's public advocate urged Mayor Rudolph Giuliani to sue the tobacco industry to recoup health care costs of smokers. 
            The Rogers case, similar to hundreds that have been filed across the country, comes on the heels of a one earlier this month in Jacksonville, Florida, where a jury awarded $750,000 to a man who smoked for 44 years before he was stricken with lung cancer.
            Corp. is a unit of Britain's B.A.T Industries Plc.In the Indianapolis case, which was tried once before and wound up in a hung jury, the Rogers family contended that the industry peddled an addictive product that was the cause of his lung cancer.Rogers originally filed the suit himself".
            Based on the information about the topic above, the label for the document is "corporate/industrial".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "corporate/industrial" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
        
    elif name == 'Reuters-21578':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            example_prompt = """
            The following is a news acticle from the Reuters financial newswire service. 
            "ending February 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the first quarter of fiscal 1986. The company said any decline would be due to expenses related to the acquisitions 
            in the middle of the current quarter of seven licensees of Sealy Inc. Because of these acquisitions, it said, first quarter sales will be substantially higher than last year's 67.1 mln dlrs." 
            Based on the information about the topic above, the label for the document is "earn".
            and the final output expected is only a list in one line. Please only return the labels with the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            [label] "earn" [/label].<|eot_id|><|start_header_id|>user<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            example_prompt = """
            The following is a news acticle from the Reuters financial newswire service. 
            "ending February 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the first quarter of fiscal 1986. The company said any decline would be due to expenses related to the acquisitions 
            in the middle of the current quarter of seven licensees of Sealy Inc. Because of these acquisitions, it said, first quarter sales will be substantially higher than last year's 67.1 mln dlrs." 
            Based on the information about the topic above, the label for the document is "earn".
            and the final output expected is only a list in one line. Please only return the labels with the format below:
            [/INST] [label] "earn" [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    else:
        raise NotImplementedError(f"Name {name} not found in the dataset list")

    return example_prompt


def get_final_main_prompt(name, model):
    if name == "AAPD":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more subjects that best describe the paper abstract. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [label] and [/label].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more subjects that best describe the paper abstract. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    elif name == "Amazon-531":
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the reviewed product. Please find exactly one label for the example. 
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [label] and [/label].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the reviewed product. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
        
    elif name == 'DBPedia-298':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more categories that best describe the Wikipedia fact. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more categories that best describe the Wikipedia fact. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] four labels [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    elif name == 'RCV1':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the newswire acticle. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [label] and [/label].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the newswire acticle. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    elif name == 'Reuters-21578':
        if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the news acticle. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        elif model == 'meta-llama/Llama-2-13b-chat-hf':
            main_prompt = """
            [INST]
            Based on the example above. Treat this as a document and assign one or more topics that best describe the news acticle. Please find exactly one label for the example.
            [DOCUMENTS]
            Please only return the labels in one line using the format below:
            [/INST] [label] and [/label].
            """
        else:
            raise NotImplementedError(f"Model {model} not found in the model list")
    else:
        raise NotImplementedError(f"Name {name} not found in the dataset list")
    
    return main_prompt