from typing import List
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.utils import urlopen
from utils.paper import Paper

class NeurIPS_2021(object):
    def __init__(self):
        self.__root_url = ''

    def collect(self):
        '''collect all papers
        
        Returns:
            list of dict(title, url, authors)
        '''
        url = 'https://papers.nips.cc/paper/2021'

        try:
            response = urlopen(url, timeout=5.0, retry=5)
        except:
            print('failed to get papers')
            return []

        soup = BeautifulSoup(response, 'html.parser')
        papers = []
        li_all = [li for li in soup.find_all('li')]
        for li in tqdm(li_all, leave=False):
            children = [c for c in li.children]
            if len(children) != 3: continue
            a = children[0]
            if a.name != 'a': continue
            i = children[2]
            if i.name != 'i': continue

            title = a.text
            paper_url = a.attrs['href']
            authors = [author.strip() for author in i.text.split(',')]
            papers.append({
                'title': title,
                'url': paper_url,
                'authors': authors,
            })
        return papers

    def get_detail(self, paper:dict):
        '''get details of the paper
        
        Args:
            paper (dict): one of the outputs from collect_papers()

        Returns:
            dictionaly which is filled with paper details
        '''
        try:
            response = urlopen(f'https://papers.nips.cc/{paper["url"]}', timeout=5.0, retry=5)
        except:
            print('failed to get paper details.')
            return paper

        soup = BeautifulSoup(response, 'html.parser')
        abstract_header = soup.find('h4', string='Abstract')
        abstract_header.select_one('p+p')
        review_url = soup.find('div', attrs={'class': 'col'}).find('a', attrs={'target': '_blank'}).attrs['href']

        try:
            response = urlopen(review_url, timeout=5.0, retry=5)
        except:
            print('failed to get paper details.')
            return paper

        soup = BeautifulSoup(response, 'html.parser')

        paper['pdf_url'] = 'https://openreview.net' + soup.find('a', attrs={'class': 'note_content_pdf'}).attrs['href']

        for notecontent in soup.find('main', attrs={'id': 'content'}).find('div', attrs={'class': 'note'}).find_all('li'):
            title = notecontent.find('strong', attrs={'class': 'note-content-field'}).text.replace(':', '').strip().lower()
            note = notecontent.find('span', attrs={'class': 'note-content-value'})
            if note.find('a') is not None:
                note = 'https://openreview.net' + note.find('a').attrs['href']
            else:
                note = note.text.strip()

            if title == 'keyword':
                paper[title] = [i.strip() for i in note.split(',')]
            else:
                paper[title] = note

        for span in soup.find_all('span', attrs={'class': 'item'}):
            if span.text.startswith('NeurIPS'):
                paper['grade'] = span.text.strip()
                break

        paper['date'] = soup.find('span', attrs={'class': 'date item'}).text.strip()

        return paper

    @classmethod
    def to_papers(cls, papers:list) -> List[Paper]:
        res = []
        for paper in papers:
            title = paper['title']
            summary = paper['abstract']
            keywords = paper['keywords']
            authors = paper['authors']
            pdf_url = paper['pdf_url']
            res.append(Paper(title, summary, keywords, authors, pdf_url))
        return res
