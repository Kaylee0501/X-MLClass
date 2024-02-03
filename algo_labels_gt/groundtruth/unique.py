#for reu
#reuters.categories() 90

#acq -> acquisition
#veg-oil -> vegetable-oil
#lin-oil -> linseed-oil
#soy-oil -> soybean oil
#rape-oil -> rapeseed-oil
#sun-oil -> sunflower-oil
#pet-chem -> petrochemical 
#nat-gas -> natural gas
#alum -> aluminum
#l-cattle -> live cattle
#instal-debt -> installment debt

#dlr -> dollar
#nzdlr -> New Zealand Dollar
#nkr -> Norwegian Krone
#dmk -> Deutsche Mark

#bop -> balance of payments
#gnp -> Gross National Product
#cpi -> Consumer Price Index
#ipi- > Industrial Production Index
#wpi -> Wholesale Price Index
#cpu -> Central Processing Unit

file_path = "./Reuters_train_labels.txt"
# Create a set to store unique words
unique_words = set()
total_count = 0
with open(file_path, 'r') as file:
    for line in file:
        # Split each line into words
        words = line.split()

        for word in words:
            #only add uniques words
            total_count += 1
            unique_words.add(word)

word_mapping = {
    "acq": "acquisition",
    "veg-oil": "vegetable oil",
    "lin-oil": "linseed oil",
    "soy-oil": "soybean oil",
    "rape-oil": "rapeseed oil",
    "sun-oil": "sunflower oil",
    "pet-chem": "petrochemical",
    "nat-gas": "natural gas",
    "alum": "aluminum",
    "l-cattle": "live cattle",
    "instal-debt": "installment debt",
    "dlr": "dollar",
    "nzdlr": "New Zealand Dollar",
    "nkr": "Norwegian Krone",
    "dmk": "Deutsche Mark",
    "bop": "balance of payments",
    "gnp": "Gross National Product",
    "cpi": "Consumer Price Index",
    "ipi": "Industrial Production Index",
    "wpi": "Wholesale Price Index",
    "cpu": "Central Processing Unit"
}

for old_word, new_word in word_mapping.items():
    if old_word in unique_words:
        unique_words.remove(old_word)
        unique_words.add(new_word)

# Convert the set back to a list
unique_words_list = list(unique_words)



print("unique", unique_words_list)
print("total count is", total_count)
print( "unique count is", len(unique_words_list))