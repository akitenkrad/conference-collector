from typing import List, Tuple, Any
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models.ldamodel import CoherenceModel

from utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class LDA(BaseEstimator):
    def __init__(self, n_topics:int=10, alpha:float=0.01, coherence='u_mass', metrics='perplexity'):
        '''Latent Dirichlet Allocation
        
        Args:
            n_topics: number of topics
            alpha: alpha of LDA model
            coherence: method for CoherenceModel (u_mass/c_v/c_uci)
        '''
        super().__init__()

        assert coherence in ['u_mass', 'c_v', 'c_uci'], f"Invalid coherence: {coherence}. coherence = ['u_mass', 'c_v', 'c_uvi']"
        assert metrics in ['perplexity', 'coherence', 'hybrid'], f"Invalid metrics: {metrics}. metrics = ['perplexity', 'coherence', 'hybrid']"

        self.n_topics = n_topics
        self.alpha = alpha
        self.coherence = coherence
        self.metrics = metrics
        self.dictionary:Dictionary = None
        self.lda:LdaModel = None

    def __str__(self):
        return f'<LDA n_topics={self.n_topics} alpha={self.alpha} coherence={self.coherence} metrics={self.metrics}>'
    def __repr__(self):
        return self.__str__()

    @classmethod
    def tokenize(cls, text:str) -> List[str]:
        words = nltk.word_tokenize(text.lower())
        words = nltk.pos_tag(words)
        res_words = []
        for word, tag in words:
            if tag[0] == 'N':
                res_words.append(word)
        return res_words

    def preprocess(self, texts:List[str]) -> List[str]:
        return [LDA.tokenize(text) for text in tqdm(texts, desc='tokenize...', leave=False)]

    def get_corpus(self, texts:List[str]) -> List[Tuple[int, int]]:
        self.dictionary = Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        return corpus

    def get_params(self, *args, **kwargs):
        return {
            'n_topics': self.n_topics,
            'alpha': self.alpha,
            'coherence': self.coherence,
        }
    
    def set_params(self, *args, **params):
        for name, value in params.items():
            if name in ['n_topics', 'alpha', 'coherence']:
                setattr(self, name, value)
        return self

    def fit(self, texts:List[str]):
        corpus = self.get_corpus(texts)
        self.lda = LdaModel(corpus=corpus, num_topics=self.n_topics, alpha=self.alpha, id2word=self.dictionary)
        return self
    
    def transform(self ,text:List[str]) -> Tuple[List[Tuple[int, float]], int, float]:
        '''apply LDA

        Args:
            text (str)

        Returns:
            topics (List[Tuple[int, float]]): topic and score
            main_topic (int): main topic
            score (float): score of main topic
        '''
        topics = self.lda[self.dictionary.doc2bow(text)]
        main_topic = max(topics, key=lambda x: x[1])[0]
        score = max(topics, key=lambda x: x[1])[0]
        return topics, main_topic, score

    def predict(self, texts:List[List[str]]) -> Tuple[List[Tuple[int, float]], int, float]:
        topics, main_topic, score = [], [], []
        for text in texts:
            _topics, _main_topic, _score = self.transform(text)
            topics.append(_topics)
            main_topic.append(_main_topic)
            score.append(_score)
        return topics, main_topic, score

    def score(self, texts:List[List[str]]) -> Tuple[float, float]:
        corpus = self.get_corpus(texts)

        if self.metrics == 'perplexity':
            log_perplexity = np.exp2(self.lda.log_perplexity(corpus))
            return log_perplexity
        elif self.metrics == 'coherence':
            cm = CoherenceModel(model=self.lda, corpus=corpus, coherence=self.coherence)
            coherence = cm.get_coherence()
            return coherence
        elif self.metrics == 'hybrid':
            log_perplexity = np.exp2(self.lda.log_perplexity(corpus))
            cm = CoherenceModel(model=self.lda, corpus=corpus, coherence=self.coherence)
            coherence = cm.get_coherence()
            return (log_perplexity + coherence) / 2.0
