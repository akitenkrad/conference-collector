from typing import List, Dict, Tuple
from copy import deepcopy

class Paper(object):
    def __init__(self, title:str, authors:List[str], summary:str, keywords:List[str], pdf_url:str):
        self.__title = title
        self.__summary = summary
        self.__authors = authors
        self.__keywords = keywords
        self.__pdf_url = pdf_url
        self.__topic = -1
        self.__score = -1.0
        self.__scores = {}
    
    def __str__(self):
        return f'<Paper "{self.title[:15]}...">'
    def __repr__(self):
        return self.__str__()

    @property
    def title(self) -> str:
        return self.__title
    @property
    def summary(self) -> str:
        return self.__summary
    @property
    def authors(self) -> List[str]:
        return deepcopy(self.__authors)
    @property
    def keywords(self) -> List[str]:
        return deepcopy(self.__keywords)
    @property
    def pdf_url(self) -> str:
        return self.__pdf_url
    @property
    def topic(self) -> int:
        return self.__topic
    @property
    def scores(self) -> Dict[int, float]:
        return self.__scores

    def set_topic(self, topic:int, score:float, scores:List[Tuple[int, float]]):
        self.__topic = topic
        self.__score = score
        self.__scores = {score[0]: score[1] for score in scores}
