from typing import List, Tuple
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class LDA(object):
    def __init__(self, texts:List[str], num_topics=10):
        self.num_topics = num_topics
        texts = [LDA.tokenize(text) for text in texts]
        self.dictionary = Dictionary(texts)
        corpus = self.get_corpus(texts)
        self.lda = LdaModel(corpus=corpus, num_topics=self.num_topics, id2word=self.dictionary)

    def get_corpus(self, texts:List[str]):
        texts = [LDA.tokenize(text) for text in texts]
        return [self.dictionary.doc2bow(text) for text in self.texts]

    @classmethod
    def tokenize(cls, text:str) -> List[str]:
        words = nltk.word_tokenize(text.lower())
        words = nltk.pos_tag(words)
        res_words = []
        for word, tag in words:
            if tag[0] == 'N':
                res_words.append(word)
        return res_words

    def transform(self ,text:str) -> Tuple[List[Tuple[int, float]], int, float]:
        '''apply LDA

        Args:
            text (str)

        Returns:
            topics (List[Tuple[int, float]]): topic and score
            main_topic (int): main topic
            score (float): score of main topic
        '''
        words = LDA.tokenize(text)
        topics = self.lda[words]
        main_topic = max(topics, key=lambda x: x[1])[0]
        score = max(topics, key=lambda x: x[1])[0]
        return topics, main_topic, score
