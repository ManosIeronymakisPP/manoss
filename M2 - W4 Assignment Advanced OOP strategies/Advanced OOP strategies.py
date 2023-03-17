import re
from collections import Counter
import string
from typing import List, Dict
'''This program defines a Python class called "UserData".
The class has a constructor method named __init__ which takes two optional arguments,
"data" and "filepath".'''

'''When, the __init__ method is called, it first assigns the "data" and "filepath" arguments.
Then it checks if a "filepath" argument was provided.
The method opens the file at that file path and reads its contents into the "data" object variable.
If a "data" argument was provided, the method assigns it to the "data" object variable.
If neither a "filepath" nor a "data" argument is provided, the constructor raises a ValueError.'''

class UserData:
    def __init__(self, data: List[str] = None, filepath: str = None):
        self.data = data
        self.filepath = filepath

        if filepath:
            with open(filepath, 'r') as f:
                self.data = f.readlines()
        elif data:
            self.data = data
        else:
            raise ValueError("Provide either data or filepath")

    '''The class has a property num_sentences which returns the number of sentences in the data.'''
    
    def num_sentences(self)-> int:
        return len(self.data)
    
    '''The class has a helper method called __count_words_in_sentence which takes one sentence as an input and returns a dictionary containing the frequency of each unique word in that sentence.
    This method uses python's re module to find all words in the sentence and python's collections module's Counter to count the frequency of each word.'''

    def __count_words_in_sentence(self, sentence: str)-> Dict[str, int]:
        words = re.findall(r'\b[a-zA-Z]+\b', sentence)
        return dict(Counter(words))
    
    '''The class has another method called count_words_in_corpus which returns the frequency of each unique word in the whole corpus,
    by calling the helper function for every sentence and combining the word count for the whole corpus.'''

    def count_words_in_corpus(self)-> Dict[str, int]:
        word_count = {}
        for sentence in self.data:
            sentence_word_count = self.__count_words_in_sentence(sentence)
            for word, count in sentence_word_count.items():
                if word in word_count:
                    word_count[word] += count
                else:
                    word_count[word] = count
        return word_count

    '''Each sentence in the corpus is passed to the helper method __count_words_in_sentence and it returns a dictionary with words and its frequency in that sentence.
    Then the result of the helper method is added to the word_count dictionary.'''
