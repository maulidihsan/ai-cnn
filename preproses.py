import os
import numpy as np
import re
import itertools
from collections import Counter

def clean_string(string):
    string = string.strip()
    string = re.sub(r'^"|"$', '', string)
    # mengganti karakter ilegal dengan spasi
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # menambah prefix spasi pada 's
    string = re.sub(r"\'s", " \'s", string)
    # menambah prefix spasi pada 've
    string = re.sub(r"\'ve", " \'ve", string)
    # menambah prefix spasi pada n't
    string = re.sub(r"n\'t", " n\'t", string)
    # menambah prefix spasi pada 're
    string = re.sub(r"\'re", " \'re", string)
    # menambah prefix spasi pada 'd
    string = re.sub(r"\'d", " \'d", string)
    # menambah prefix spasi pada 'll
    string = re.sub(r"\'ll", " \'ll", string)
    # beri spasi pada tanda baca
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # fix tanda kurung dan tanda tanya
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() # strip outer space and lower them

def read_data_bersih(path):
    documents = list()
    # Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open('%s/%s' % (path, filename)) as f:
                doc = f.read()
                doc = clean_string(doc)
                documents.append(doc)
    
    # Read in all lines in a txt file
    if os.path.isfile(path):        
        with open(path) as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_string(line))
    return documents

def load_data(data_positif, data_negatif):
    # load data dari files
    # =====================
    sampel_positif = read_data_bersih(data_positif)
    sampel_negatif = read_data_bersih(data_negatif)
    
    # append kedua sampel
    x_text = sampel_positif + sampel_negatif

    # Beri label
    positive_labels = [[0, 1] for _ in sampel_positif]
    negative_labels = [[1, 0] for _ in sampel_negatif]

    # concat jadi sebuah matriks
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iterator(data, batch_size, jumlah_epochs, acak=True):
    data = np.array(data)
    data_size = len(data)
    batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(jumlah_epochs):
        # Shuffle data pada tiap epoch
        #print("Current epochs: %i" % epoch)
        print
        if acak:
            shuffle_index = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_index]
        else:
            shuffled_data = data
        for batch_num in range(batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]