import numpy as np
import re

data = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance."""
# The original data variable in the source is split across multiple lines and 'Out' cells
# The content is reconstructed here.

sentences = data.split('.')
# The output for sentences is:
# ['Deep learning (also known as deep structured learning) is part of a broader\n family of machine learning methods based on artificial neural networks with r\n epresentation learning', 
# 'Learning can be supervised, semi-supervised or unsupe\n rvised', 
# 'Deep-learning architectures such as deep neural networks, deep belief\n networks, deep reinforcement learning, recurrent neural networks, convolution\n al neural networks and Transformers have been applied to fields including com\n puter vision, speech recognition, natural language processing, machine transl\n ation, bioinformatics, drug design, medical image analysis, climate science,\n material inspection and board game programs, where they have produced results\n comparable to and in some cases surpassing human expert performance', 
# '']

clean_sent = []
for sentence in sentences:
    if sentence == "":
        continue
    # The first re.sub in the source is missing an assignment, assuming it's meant to clean special characters
    sentence = re.sub('[^A-Za-z0-9]+', ' ', (sentence))
    # The second re.sub seems to be intended to remove single characters that are surrounded by spaces or start of string and a space
    # The regex r'(?:^| )\w(?:\$ )' in the source looks unusual. Based on typical preprocessing, a common pattern to remove isolated single characters is r'\b\w\b'.
    # I will use the literal expression from the source, assuming the $ sign is a typo for a space or a problem with transcription.
    # Given the output in the source, it seems this line may have had no effect or was intended differently.
    # The clean_sent output suggests only replacing non-alphanumeric with spaces, lowercasing, and stripping was done.
    # Replicating the code literally:
    sentence = re.sub(r'(?:^| )\w(?:\$ )', '', (sentence)).strip()
    sentence = sentence.lower()
    clean_sent.append(sentence)

# The output for clean_sent is:
# ['deep learning also known as deep structured learning is part of broader fam ily of machine learning methods based on artificial neural networks with repr esentation learning', 
# 'learning can be supervised semi supervised or unsupervised', 
# 'deep learning architectures such as deep neural networks deep belief networ ks deep reinforcement learning recurrent neural networks convolutional neural networks and transformers have been applied to fields including computer visi on speech recognition natural language processing machine translation bioinfo rmatics drug design medical image analysis climate science material inspectio n and board game programs where they have produced results comparable to and in some cases surpassing human expert performance']

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_sent)
sequences = tokenizer.texts_to_sequences(clean_sent)
print(sequences)
# The output for sequences is printed across multiple lines in the source.

index_to_word = {}
word_to_index = {}
for i, sequence in enumerate(sequences):
    word_in_sentence = clean_sent[i].split()
    # print(sequence) # code from source
    # print(word_in_sentence) # code from source
    for j, value in enumerate(sequence):
        index_to_word[value] = word_in_sentence[j]
        word_to_index[word_in_sentence[j]] = value
# print(index_to_word, "\n") # code from source
# print(word_to_index) # code from source

vocab_size = len(tokenizer.word_index) + 1
emb_size = 10
context_size = 2
contexts = []
targets = []
for sequence in sequences:
    # This loop logic is for Continuous Bag-of-Words (CBOW)
    for i in range(context_size, len(sequence) - context_size):
        target = sequence[i]
        context = [sequence[i - 2], sequence[i - 1], sequence[i + 1], sequence[i + 2]]
        # print(context) # code from source
        contexts.append(context)
        targets.append(target)
# print(contexts, "\n") # code from source
# print(targets) # code from source

# printing features with target (just showing the first 5 examples)
for i in range(5):
    words = []
    target = index_to_word.get(targets[i])
    for j in contexts[i]:
        words.append(index_to_word.get(j))
    # print(words, " -> ", target) # code from source

X = np.array(contexts)
Y = np.array(targets)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=2*context_size),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# The fit call is only shown, not the definition of history itself.
# history = model.fit(X, Y, epochs=80) 

from sklearn.decomposition import PCA
# The fit was performed with history = model.fit(X, Y, epochs=80)
embeddings = model.get_weights()[0]
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Test sentences for prediction
test_sentenses = [
    "known as structured learning",
    "transformers have applied to",
    "where they produced results",
    "cases surpassing expert performance"
]

# Prediction loop
for sent in test_sentenses:
    test_words = sent.split(" ")
    # print(test_words) # code from source
    x_test = []
    for i in test_words:
        # word_to_index.get(i) might return None if a word isn't in the vocabulary.
        # This implementation assumes all words are present.
        x_test.append(word_to_index.get(i))
    x_test = np.array([x_test])
    # print(x_test) # code from source
    
    pred = model.predict(x_test)
    pred = np.argmax(pred[0])
    # print("pred", test_words, "\n=", index_to_word.get(pred), "\n\n") # code from source

# The final code snippet shows plotting the history, which would require the 'history' object
# import seaborn as sns
# sns.lineplot(model.history.history)
