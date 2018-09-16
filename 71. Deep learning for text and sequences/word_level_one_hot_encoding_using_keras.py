from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Create a tokenizer to take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)
# Builds the word index
tokenizer.fit_on_texts(samples)
# turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)
# One-hot encode
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# Recover the word index that was computed
word_index = tokenizer.word_index
print(word_index)
