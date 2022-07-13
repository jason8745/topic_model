import pickle
import gensim
import os



class LdaModel():
    def __init__(self) -> None:
        self.dictionary = None
        self.corpus = None
        self.model = None
        
    




    def train_model(self,text_data,topic_count):
        """
        :param int topic_count: lda topic count
        """
        self.dictionary = gensim.corpora.Dictionary(text_data )
        self.corpus = [self.dictionary.doc2bow(text) for text in text_data]
        self.model = gensim.models.ldamodel.LdaModel(self.corpus, num_topics = topic_count, id2word=self.dictionary, passes=15)
        print('train model  success')

    
    def _save_lda_corpus(self,name:str):
        if  self.corpus !=None:
            pickle.dump(self.corpus, open(f'./model/{name}/corpus.pkl', 'wb'))
        else:
            print('No corpus')
    
    def _save_lda_dict(self,name:str):
        if self.dictionary !=None:
            self.dictionary.save(f'./model/{name}/dictionary.gensim')
        else:
            print('No dictionary')

    def _save_lda_model(self,name:str):
        if self.model !=None:
            self.model.save(f'./model/{name}/ldamodel.gensim')
        else:
            print('No model')

    def save_lda(self,name):
        if not os.path.isdir(f'./model/{name}'):
            os.makedirs(f'./model/{name}')
        self._save_lda_corpus(name)
        self._save_lda_dict(name)
        self._save_lda_model(name)
        print(f'save lda object at ./model/{name}')

    def load_lda(self,name):
        self.dictionary = gensim.corpora.Dictionary.load(f'./model/{name}/dictionary.gensim')
        self.corpus = pickle.load(open(f'./model/{name}/corpus.pkl', 'rb'))
        self.model = gensim.models.ldamodel.LdaModel.load(f'./model/{name}/ldamodel.gensim')
        print(f'load model :{name} success')




    def predict(self,text)->list:
        """
        text must be preprocess
        """
        new_doc_bow = self.dictionary.doc2bow(text)
        result = self.model.get_document_topics(new_doc_bow)
        for res in result:
            print('===========')
            print(f'{round(100*res[1],2)} % belongs to topic: {res[0]}')

        return result
