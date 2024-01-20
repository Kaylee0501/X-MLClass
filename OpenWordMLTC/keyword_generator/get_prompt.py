import fire

def create_prompt(name):
    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics.
    <</SYS>>
    """
    example_prompt = example_prompt(name)

    main_prompt = """
    [INST]
    I have a topic that contains the following documents:
    [DOCUMENTS]

    Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:
    [/INST]
    """

    prompt = system_prompt + example_prompt + main_prompt 

    return prompt


def example_prompt(name):
    if name == "AAPD":
        example_prompt = """
        I have a topic that contains the following documents:
        - the relation between pearson 's correlation coefficient and salton 's cosine measure is revealed based on the different possible values of the division of the l1 norm and the l2 norm of a vector 
        - these different values yield a sheaf of increasingly straight lines which form together a cloud of points , being the investigated relation the theoretical results are tested against the author co citation relations 
        - among 24 informetricians for whom two matrices can be constructed , based on co citations the asymmetric occurrence matrix and the symmetric co citation matrix both examples completely confirm the theoretical results 
        - the results enable us to specify an algorithm which provides a threshold value for the cosine above which none of the corresponding pearson correlations would be negative using this threshold value can be expected to optimize the visualization of the vector space"

        Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:

        [/INST] Information Retrieval, Margin of Error
        """
    elif name == "Amazon-531":
        example_prompt = """
        I have a topic that contains the following documents:
        - omron hem 790it automatic blood pressure monitor with advanced omron health management software so far this machine has worked well
        - and is very simple to use . it is nice to have immediate feedback on the bloodpressure effects of my various exercises , 
        - food consumption , and relaxation or stress levels .

        Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:

        [/INST] health_personal_care, medical_supplies_equipment, health_monitors
        """
    elif name == 'DBPeida-298':
        example_prompt = """
        I have a topic that contains the following documents:
        - the singing cisticola ( cisticola cantans ) is a species of bird in the cisticolidae family . it is found in benin , 
        - burkina faso , burundi , cameroon , central african republic , chad , democratic republic of the congo , ivory coast , eritrea , 
        - ethiopia , gambia , ghana , guinea , guinea bissau , kenya , liberia , malawi , mali , mozambique , niger , nigeria , rwanda , 
        - senegal , sierra leone , sudan , tanzania , togo , uganda , zambia , and zimbabwe . its natural habitats are subtropical or tropical 
        - dry forests and subtropical or tropical dry shrubland

        Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:

        [/INST] species, animal, bird
        """
    elif name ==  'RCV1-V2':
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
    elif name == 'Reuters-21578':
        example_prompt = """
        I have a topic that contains the following documents:
        - OHIO MATTRESS &lt;OMT> MAY HAVE LOWER 1ST QTR NET Ohio Mattress Co said its first quarter, ending February 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the first quarter of fiscal
        - 1986. The company said any decline would be due to expenses related to the acquisitions in the middle of the current quarter of seven licensees of Sealy Inc, as well as 82 pct of
        - the outstanding capital stock of Sealy. Because of these acquisitions, it said, first quarter sales will be substantially higher than last year's 67.1 mln dlrs. Noting that it typically reports first quarter results in
        - late march, said the report is likely to be issued in early April this year. It said the delay is due to administrative considerations, including conducting appraisals, in connection with the acquisitions.
        
        Based on the information about the topic above, please find at most three labels for this topic above. Please output your answer use the following format in one line:

        [/INST] acquisitions, earnings
        """
    else:
        example_prompt = None

    return example_prompt