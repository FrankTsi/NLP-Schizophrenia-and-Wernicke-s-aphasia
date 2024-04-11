# importing nltk to remove stop words from text
import nltk
import re
nltk.download('stopwords')
stop_words=stopwords.words('english')

# creating a list of fillers and puncutation remarks that I wanted to make sure are removed

punctuation_list = [".",",","?","!",'"',"\n"]
fillers = ["Yes","yes","yeah","okay","Okay","Yeah","Yea","yea","Ok","ok","OK","Huh",'huh',"uh",'um','aha','Uh','Um','Aha','oh','Oh',"Hm","Hmm","hm","hmm"]



# importing w2v model to get vectors for words
import gensim
W2V_PATH="/Users/anasmayya/Desktop/EMCL++/Thesis/NLP/Word2Vec/GoogleNews-vectors-negative300.bin"
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)

# importing panda
import pandas as pd

# importing spacy large english library to use for tokenization

import spacy
nlp = spacy.load("en_core_web_lg")


# defining similarity function to calculate w2v distance between words on a given window

def similarity(index, window, text): 
    if window + index < len(text): # window should be equal to or smaller than the difference between list length and current index 
        return model_w2v.similarity(text[index],text[index+window])



# Participant alone: First we worked on the participant text only to measure their internal coherence
# naming the file I analyzed. Each row consisted of a the participant answer, excluding the interviewer, on a task. So if the participant did 5 tasks, they would have 5 rows.


df = pd.read_csv("/Users/anasmayya/Desktop/EMCL++/Thesis/Data/combined data/DF_PAR_with_diagnosis_may_25.csv")


# carrying out average and variance of similarity function with windows from 1 to 20 and assigning them to a column

for window in range(1,20):
    df[f"ave_window_{window}"]= pd.Series(dtype='float64')  # create columns 
    df[f"var_window_{window}"]= pd.Series(dtype='float64') # create columns
    for index_row, row in df.iterrows(): # iterate over every row in the df column Text
        row = nlp(row["Text"]) # make the content of Text a tokenized nlp object 
        vocab_in_model = []

        for token in (row):

            token = token.text.lower() # take the string from the tokenized list and make it lower case

            if token in model_w2v.key_to_index and token not in stop_words and token not in punctuation_list and token not in fillers: # if the token is part of the vocabulary of word2vec

                vocab_in_model.append(token) # add it to the list vocab_in_model

        total_similarity = 0 # a variable to add the similarity scores to

        similarity_score_list = []

        for index, word in enumerate(vocab_in_model):

            #similarity_score = similarity(index,1,vocab_in_model) # assign the similarity score to a variable
            #total_similarity += similarity_score # add up the similarity scores
            similarity_score = similarity(index,window,vocab_in_model)
            if similarity_score is not None:
                similarity_score_list.append(similarity(index,window,vocab_in_model)) # add each similarity score to a list
        #print(vocab_in_model)
        #print(similarity_score_list) 
        if len(similarity_score_list):
            average = sum(similarity_score_list)/len(similarity_score_list) # calculate average for each row for a given window size
        df.at[index_row, f"ave_window_{window}"] = average # assign everage to the average column

        sum_of_squares = 0 # i add up the squared difference between score and average to this variable

        for item in similarity_score_list: # 
            sum_of_squares +=(item - average)**2
        if len(similarity_score_list):
            variance = sum_of_squares/len(similarity_score_list)
        df.at[index_row, f"var_window_{window}"] = variance


-----------------------------------------------------------------
# Participant and Interviwer together

# here we worked on a different dataframe that included the turns of both the interviewer and the participant. Every turn was a row, so if each spoke 50 times, we would end up with 100 rows.


from sklearn.metrics.pairwise import cosine_similarity

# importing the data frame
df = pd.read_csv("/Users/anasmayya/Desktop/EMCL++/Thesis/Data/combined data/concatenated PAR INV data/DF_PAR_INV_concatenated_MAY_9.csv")


# we wanted to (1) compare average similarity of participant with average similarity of interviwer previous turn, (2) average similairty of first sentence of participant with average similarity of interviewer previous turn, (3) average similairty of second sentence of participant with average similarity of interviewer previous turn, (4) average similairty of third sentence of participant with average similarity of interviewer previous turn

df["INT_PAR_distance_score"] = pd.Series(dtype='float64')  # create an empty column for total words distance
df["INT_PAR_2_8"] = pd.Series(dtype='float64')  # create an empty column for 2-8 words distance
df["INT_PAR_8_14"] = pd.Series(dtype='float64')  # create an empty column for 2-8 words distance
df["INT_PAR_14_20"] = pd.Series(dtype='float64')  # create an empty column for 2-8 words distance

INT_avg = None  # start with a None average for INT
for index, row in df.iterrows():  # loop over the rows of the dataframe
    if row["Speaker"] == "INV":  # check if the speaker is the interviewer
        text = nlp(row["Text"])  # tokenize the text
        INT_word_vectors = []  # create an empty list to append word vectors to

        for token in text:  # loop over the tokens in the text
            token = token.text.lower()  # lower them and get their string

            if token in model_w2v.key_to_index and token not in stop_words and token not in fillers and token not in punctuation_list:
                # check that the tokens are in the model, not stop words, not punctuation, not fillers
                INT_word_vector = model_w2v[token]  # get the vector of each token
                INT_word_vectors.append(INT_word_vector)  # append the vector to a list

        if INT_word_vectors:  # if the Int word vector is not empty
            INT_avg = np.mean(INT_word_vectors, axis=0)  # average it and assign it to INT_AVG which was NONE
        continue

    elif row["Speaker"] == "PAR" and row["ID"] == df.iloc[index - 1]["ID"] and row["Task"] == df.iloc[index - 1]["Task"]:
        # if the speaker is par, and if the ID and task match with the previous column
        text = nlp(row["Text"])  # tokenize text
        PAR_word_vectors = []  # # create an empty list to append word vectors to

        for token in text:  # loop over the tokens in the text
            token = token.text.lower()  # lower them and get their string

            if token in model_w2v.key_to_index and token not in stop_words and token not in fillers and token not in punctuation_list:
                PAR_word_vector = model_w2v[token]
                PAR_word_vectors.append(PAR_word_vector)

        if PAR_word_vectors and INT_avg is not None:
            PAR_avg = np.mean(PAR_word_vectors, axis=0)
            df.at[index, "INT_PAR_distance_score"] = cosine_similarity(INT_avg.reshape(1, -1),
                                                                       PAR_avg.reshape(1, -1))[0][0]

        if len(PAR_word_vectors) >= 19:
            PAR_avg_20_words = np.mean(PAR_word_vectors[13:20], axis=0)
            if INT_avg is not None:
                df.at[index, "INT_PAR_14_20"] = cosine_similarity(INT_avg.reshape(1, -1),
                                                                   PAR_avg_20_words.reshape(1, -1))[0][0]

        if len(PAR_word_vectors) >= 13:
            PAR_avg_14_words = np.mean(PAR_word_vectors[7:14], axis=0)
            if INT_avg is not None:
                df.at[index, "INT_PAR_8_14"] = cosine_similarity(INT_avg.reshape(1, -1),
                                                                 PAR_avg_14_words.reshape(1, -1))[0][0]
        if len(PAR_word_vectors) >= 7:
            PAR_avg_8_words = np.mean(PAR_word_vectors[1:8], axis=0)
            if INT_avg is not None:
                df.at[index, "INT_PAR_2_8"] = cosine_similarity(INT_avg.reshape(1, -1),
                                                                PAR_avg_8_words.reshape(1, -1))[0][0]