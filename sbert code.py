
df = pd.read_csv("/Users/anasmayya/Desktop/EMCL++/Thesis/Data/aphasia bank healthy/aphasia_bank_healthy_par_only_may_25.csv")



!pip install transformers sentencepiece datasets sentence-transformers nltk

# importing libraries and functions

import transformers
import torch
import pandas as pd
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer


from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer



tokenizer = RegexpTokenizer(r'\w+')



# defining the similarity function, which calculates the distance between sentences on a given window

def similarity(index, window, text):
    if window + index < len(text):
        return cosine_similarity(text[index].reshape(1, -1), text[index + window].reshape(1, -1))

# assigning tokenizer name


tokenizer = RegexpTokenizer(r'\w+')


# creating columns for averages and varainces for range from 1 to 3
for window in range(1,4): 
    df[f"sbert_ave_window_{window}"]= pd.Series(dtype='float64')
    df[f"sbert_var_window_{window}"]= pd.Series(dtype='float64')



# iterating over rows to calculate the scores 

    for index, row in df.iterrows(): # iterate over df rows
        sentence_embedding_list = [] # create empty list to append embedding scores to
        sentences = re.split('[.!?]',(row["Text"])) # split sentence with the following separators
        sentence_list = [] # create list to append sentences to
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) > 2:
                sentence_list.append(sentence) # only append sentences who have more than three words
        for sentence in sentence_list:
            sentence_embedding = model.encode(sentence)
            if sentence_embedding is not None:
                sentence_embedding_list.append(sentence_embedding)
        
        similarity_score_list = []
        for num, sentence in enumerate(sentence_embedding_list):
            similarity_score = similarity(num, window, sentence_embedding_list)
            if similarity_score is not None:
                similarity_score_list.append(similarity_score) # add each similarity score to a list
        
        if len(similarity_score_list):
            print(index)
            df.at[index, f"sbert_ave_window_{window}"] = np.average(similarity_score_list)
            df.at[index, f"sbert_var_window_{window}"] = np.var(similarity_score_list)






----------------------------------------------------

# Participant and interviewer
# we work here on the file that included both interviewer and participant speech, every row included a turn. So if the participant had 50 turns and the interviewer spoke 50 turns, we would have 100 rows.

df = pd.read_csv("/Users/anasmayya/Desktop/EMCL++/Thesis/Data/aphasia bank healthy/aphasia_bank_healthy_par_int_may_25.csv")




import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# import tokenizer

tokenizer = RegexpTokenizer(r'\w+')

# import transformer model

model = SentenceTransformer(model_name, device='cpu')

# here, for every turn of speaking, first we calculate the average similarity score of the interviewer, then the similarity distance between this average, and the first sentence the participant next says, the second sentence the participant next says, the third sentence the participant next says. We only compare turns on the same task

INT_avg = None
prev_row = None  # Store the previous row
for index, row in df.iterrows():
    if row["Speaker"] == "INV":
        sentence_embedding_list = []
        sentences = re.split('[.!?]', row["Text"])
        sentence_list = []
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) > 2:
                sentence_list.append(sentence)
        for sentence in sentence_list:
            sentence_embedding = model.encode(sentence)
            if sentence_embedding is not None:
                sentence_embedding_list.append(sentence_embedding)
        if len(sentence_embedding_list):
            INT_avg = np.average(sentence_embedding_list, axis=0)
        prev_row = row  # Update the previous row
        continue

    elif row["Speaker"] == "PAR" and prev_row is not None and row["ID"] == prev_row["ID"] and row["Task"] == prev_row["Task"]:
        sentence_embedding_list = []
        sentences = re.split('[.!?]', row["Text"])
        sentence_list = []
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) > 2:
                sentence_list.append(sentence)
        for sentence in sentence_list:
            sentence_embedding = model.encode(sentence)
            if sentence_embedding is not None:
                sentence_embedding_list.append(sentence_embedding)
        if len(sentence_embedding_list):
            PAR_avg = np.average(sentence_embedding_list, axis=0)
            df.at[index, "INT_PAR_sbert_distance_score"] = cosine_similarity(np.array([INT_avg]).reshape(1, -1), np.array([PAR_avg]).reshape(1, -1))[0][0]

        if len(sentence_embedding_list) >= 3:
            df.at[index, "INT_PAR_sbert3"] = cosine_similarity(np.array([INT_avg]).reshape(1, -1), np.array([sentence_embedding_list[2]]).reshape(1, -1))[0][0]
        if len(sentence_embedding_list) >= 2:
            df.at[index, "INT_PAR_sbert2"] = cosine_similarity(np.array([INT_avg]).reshape(1, -1), np.array([sentence_embedding_list[1]]).reshape(1, -1))[0][0]
        if len(sentence_embedding_list) >= 1:
            df.at[index, "INT_PAR_sbert1"] = cosine_similarity(np.array([INT_avg]).reshape(1, -1), np.array([sentence_embedding_list[0]]).reshape(1, -1))[0][0]
    prev_row = row  # Update the previous row for the next iteration
