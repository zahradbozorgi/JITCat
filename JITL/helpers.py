import os
import time
import pandas as pd
import numpy as np


def get_size_max_len(sequences):
    # Tokenize the words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    vocab_size = len(tokenizer.word_index) + 1

    # Convert words to sequences of integers
    sequences_int = tokenizer.texts_to_sequences(sequences)

    # Generate input and target sequences for training
    max_sequence_len = max([len(seq) for seq in sequences_int])
    
    return vocab_size, max_sequence_len


def tokenize_and_pad(sequences, max_sequence_len):
    # Tokenize the words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Convert words to sequences of integers
    sequences_int = tokenizer.texts_to_sequences(sequences)
    
    # Generate input and target sequences for training
    # max_sequence_len = max([len(seq) for seq in sequences_int])
    input_sequences = []
    target_sequences = []
    for seq in sequences_int:
        # print(seq)
        for i in range(1, len(seq)):
            input_seq = seq[:i]
            target_seq = seq[i]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
            
    # Pad sequences to have the same length
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    target_sequences = np.array(target_sequences)
    
    return input_sequences, target_sequences

def remove_first_row(group):
    return group.iloc[1:]

def remove_last_row(group):
    return group.iloc[:-1]
