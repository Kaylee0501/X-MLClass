import csv
def extract_top1(csv_file):
    representation_top1 = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            representation = row['Representation']
            #extracts the first element and remove the leading and trailing single quotes
            representation = representation.strip("[]").split(', ')[0][1:-1]
            representation_top1.append(representation)

    return representation_top1


# aapd_csv_file = "./labelsGenerated/AAPDtopic_info_3000.csv"
# aapdlabels = extract_top1(aapd_csv_file)
# # print("all", aapdlabels)
# print("count is", len(aapdlabels))
# print("remove duplicate is", len(set(aapdlabels)))
# if(len(aapdlabels) != len(set(aapdlabels))):
#     aapdlabels = list(set(aapdlabels))
# print(aapdlabels)
# print(len(aapdlabels))
# print("-------------------------")
# reuters_csv_file = "./labelsGenerated/Reuterstopic_info_3000.csv"
# reuterslabels = extract_top1(reuters_csv_file)
# # print("all", reuterslabels)
# print("count is", len(reuterslabels))
# print("remove duplicate is", len(set(reuterslabels)))
# if(len(reuterslabels) != len(set(reuterslabels))):
#     reuterslabels = list(set(reuterslabels))
# print(reuterslabels)
# print(len(reuterslabels))
# print("-------------------------")
# rcv_csv_file = "./labelsGenerated/RCVtopic_info_3000.csv"
# rcvlabels = extract_top1(rcv_csv_file)
# # print("all", rcvlabels)
# print("count is", len(rcvlabels))
# print("remove duplicate is", len(set(rcvlabels)))
# if(len(rcvlabels) != len(set(rcvlabels))):
#     rcvlabels = list(set(rcvlabels))
# print(rcvlabels)
# print(len(rcvlabels))
# print("-------------------------")
# amazon_csv_file = "./labelsGenerated/Amazontopic_info_14000.csv"
# amazonlabels = extract_top1(amazon_csv_file)
# # print("all", amazonlabels)
# print("count is", len(amazonlabels))
# print("remove duplicate is", len(set(amazonlabels)))
# if(len(amazonlabels) != len(set(amazonlabels))):
#     amazonlabels = list(set(amazonlabels))
# print(amazonlabels)
# print(len(amazonlabels))
# print("-------------------------")
# dbpedia_csv_file = "./labelsGenerated/DBPediatopic_info_8000.csv"
# dbpedialabels = extract_top1(dbpedia_csv_file)
# # print("all", dbpedialabels)
# print("count is", len(dbpedialabels))
# print("remove duplicate is", len(set(dbpedialabels)))
# if(len(dbpedialabels) != len(set(dbpedialabels))):
#     dbpedialabels = list(set(dbpedialabels))
# print(dbpedialabels)
# print(len(dbpedialabels))


def extract_top2(csv_file):
    representation_top2 = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            representation = row['Representation']
            representation1 = representation.strip("[]").split(', ')[0][1:-1]
            representation2 = representation.strip("[]").split(', ')[1][1:-1]
            representation_top2.append(representation1)
            representation_top2.append(representation2)
    return representation_top2


# aapd_csv_file = "./labelsGenerated/AAPDtopic_info_3000.csv"
# aapdlabels = extract_top2(aapd_csv_file)
# # print("all", aapdlabels)
# print("count is", len(aapdlabels))
# print("remove duplicate is", len(set(aapdlabels)))
# if(len(aapdlabels) != len(set(aapdlabels))):
#     aapdlabels = list(set(aapdlabels))
# print(aapdlabels)
# print(len(aapdlabels))
# print("-------------------------")
# reuters_csv_file = "./labelsGenerated/Reuterstopic_info_3000.csv"
# reuterslabels = extract_top2(reuters_csv_file)
# # print("all", reuterslabels)
# print("count is", len(reuterslabels))
# print("remove duplicate is", len(set(reuterslabels)))
# if(len(reuterslabels) != len(set(reuterslabels))):
#     reuterslabels = list(set(reuterslabels))
# print(reuterslabels)
# print(len(reuterslabels))
# print("-------------------------")
# rcv_csv_file = "./labelsGenerated/RCVtopic_info_3000.csv"
# rcvlabels = extract_top2(rcv_csv_file)
# # print("all", rcvlabels)
# print("count is", len(rcvlabels))
# print("remove duplicate is", len(set(rcvlabels)))
# if(len(rcvlabels) != len(set(rcvlabels))):
#     rcvlabels = list(set(rcvlabels))
# print(rcvlabels)
# print(len(rcvlabels))
# print("-------------------------")
# amazon_csv_file = "./labelsGenerated/Amazontopic_info_14000.csv"
# amazonlabels = extract_top2(amazon_csv_file)
# # print("all", amazonlabels)
# print("count is", len(amazonlabels))
# print("remove duplicate is", len(set(amazonlabels)))
# if(len(amazonlabels) != len(set(amazonlabels))):
#     amazonlabels = list(set(amazonlabels))
# print(amazonlabels)
# print(len(amazonlabels))
# print("-------------------------")
# dbpedia_csv_file = "./labelsGenerated/DBPediatopic_info_8000.csv"
# dbpedialabels = extract_top2(dbpedia_csv_file)
# # print("all", dbpedialabels)
# print("count is", len(dbpedialabels))
# print("remove duplicate is", len(set(dbpedialabels)))
# if(len(dbpedialabels) != len(set(dbpedialabels))):
#     dbpedialabels = list(set(dbpedialabels))
# print(dbpedialabels)
# print(len(dbpedialabels))




def extract_llama(csv_file):
    representation_llamatop1 = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            representation = row['Llama2']
            #extracts the first element and remove the leading and trailing single quotes and 
            representation = representation.strip("[]").split(', ')[0][1:-1]
            # Remove leading and trailing newline characters from each item in the list
            representation = representation.strip("\\n")
            representation_llamatop1.append(representation)

    return representation_llamatop1



def extract_top2_combined(csv_file):
    representation_top2 = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            representation = row['Representation']
            representation1 = representation.strip("[]").split(', ')[0][1:-1]
            representation2 = representation.strip("[]").split(', ')[1][1:-1]
            representation_combined = representation1 + " " + representation2
            representation_top2.append(representation_combined)
    return representation_top2

aapd_csv_file = "./raw_text_topic_info/AAPDtopic_info_3000.csv"
aapdlabels = extract_top2_combined(aapd_csv_file)
# print("all", aapdlabels)
print("count is", len(aapdlabels))
print("remove duplicate is", len(set(aapdlabels)))
if(len(aapdlabels) != len(set(aapdlabels))):
    aapdlabels = list(set(aapdlabels))
print(aapdlabels)
print(len(aapdlabels))
print("-------------------------")
reuters_csv_file = "./raw_text_topic_info/Reutertopic_info_3000.csv"
reuterslabels = extract_top2_combined(reuters_csv_file)
# print("all", reuterslabels)
print("count is", len(reuterslabels))
print("remove duplicate is", len(set(reuterslabels)))
if(len(reuterslabels) != len(set(reuterslabels))):
    reuterslabels = list(set(reuterslabels))
print(reuterslabels)
print(len(reuterslabels))
print("-------------------------")
rcv_csv_file = "./raw_text_topic_info/RCVtopic_info_3000.csv"
rcvlabels = extract_top2_combined(rcv_csv_file)
# print("all", rcvlabels)
print("count is", len(rcvlabels))
print("remove duplicate is", len(set(rcvlabels)))
if(len(rcvlabels) != len(set(rcvlabels))):
    rcvlabels = list(set(rcvlabels))
print(rcvlabels)
print(len(rcvlabels))
print("-------------------------")
amazon_csv_file = "./raw_text_topic_info/Amazontopic_info_14000.csv"
amazonlabels = extract_top2_combined(amazon_csv_file)
# print("all", amazonlabels)
print("count is", len(amazonlabels))
print("remove duplicate is", len(set(amazonlabels)))
if(len(amazonlabels) != len(set(amazonlabels))):
    amazonlabels = list(set(amazonlabels))
print(amazonlabels)
print(len(amazonlabels))
print("-------------------------")
dbpedia_csv_file = "./raw_text_topic_info/DBPtopic_info_8000.csv"
dbpedialabels = extract_top2_combined(dbpedia_csv_file)
# print("all", dbpedialabels)
print("count is", len(dbpedialabels))
print("remove duplicate is", len(set(dbpedialabels)))
if(len(dbpedialabels) != len(set(dbpedialabels))):
    dbpedialabels = list(set(dbpedialabels))
print(dbpedialabels)
print(len(dbpedialabels))