
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread


class DataPreprocess():
    def __init__(self) -> None:
        self.en_stop=None
        self.text_data=[]
        self.__load_process_dataset()
        
        

    def __load_process_dataset(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.en_stop = set(nltk.corpus.stopwords.words('english'))




    def __text_tokenize(self,text):
        parser = English()
        lda_tokens=[]
        #透過spacy的English模型 對text處理
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens



    def __lemma_transfer(self,word):
        """
        >>> print(wn.morphy('dogs'))
        dog
        >>> print(wn.morphy('churches'))
        church
        >>> print(wn.morphy('aardwolves'))
        aardwolf
        """
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma
    

    def prepare_text_for_lda(self,text):
        tokens = self.__text_tokenize(text)
        #只保留長度>4
        tokens =[token for token in tokens if len(token)>4]
        tokens =[token for token in tokens if token not in self.en_stop]
        tokens = [self.__lemma_transfer(token) for token in tokens]
        return tokens

    def add_word_to_text_data_list(self,text):
        """
        TODO:
        為了多線程操作,集成至此function
        """
        token = self.prepare_text_for_lda(text)
        if random.random()>.99:
            self.text_data.append(token)


    def transform_dataset_to_lda(self,filepath):
        # 沒有多線程要跑10m
        # 可以嘗試multi-thread
        # executor = ThreadPoolExecutor(5)
        start=time.time()
        with open(filepath) as f:
            with ThreadPoolExecutor(max_workers = 8) as executor:
                executor.map(self.add_word_to_text_data_list, f)
        end = time.time()
        print(f'success transform data,cost: {end-start}')

 
