from typing import List
import os
from os import PathLike
from pathlib import Path
from nltk import FreqDist

from utils.utils import is_notebook, word_cloud
from utils.paper import Paper
from utils.lda import LDA

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class Report(object):

    @classmethod
    def s(cls, text:str):
        return f'\n{text}  \n'
    @classmethod
    def l(cls, text:str):
        return f'\n- {text}  '

    @classmethod
    def analyze(self, papers:List[Paper], n_topics:int):
        lda = LDA([paper.title + '\n' + paper.summary for paper in papers], num_topics=n_topics)
        for paper in tqdm(papers):
            topics, main_topic, score = lda.transform(paper.title + '\n' + paper.summary)
            paper.set_topic(main_topic, score, topics)

        topics = {}
        for paper in papers:
            if paper.topic not in topics:
                topics[paper.topic] = {'papers': []}
            topics[paper.topic]['papers'].append(paper)
        return lda, topics

    @classmethod
    def report(cls, report_title:str, out_dir:PathLike, papers:List[Paper], n_topics:int=10):
        out_dir = Path(out_dir)
        (out_dir / 'images').mkdir(parents=True, exist_ok=True)
        report_file = out_dir / 'report.md'

        # analyze
        lda, topics = Report.analyze(papers, n_topics)

        with open(report_file, 'wt', encoding='utf-8') as wf:
            wf.write(Report.s(f'# {report_title}'))
            for i in tqdm(range(n_topics), total=n_topics):
                topic_papers = topics[i]['papers']

                # keywords -> wordcloud
                topics[i]['keyword_freqdist']= FreqDist()
                topics[i]['text_freqdist'] = FreqDist()
                keywords = []
                for paper in topic_papers:
                    keywords += [keyword.lower() for keyword in paper.keywords]
                    for keyword in paper.keywords:
                        topics[i]['keyword_freqdist'][keyword] += 1

                    words = LDA.tokenize(paper.title + '\n' + paper.summary)
                    for word in words:
                        topics[i]['text_freqdist'][word] += 1

                keywords = ' '.join(keywords)
                word_cloud(keywords, str(out_dir / f'images/topic.{i:02d}.keywords.png'))

                # title + abstract -> wordcloud
                texts = []
                for paper in topic_papers:
                    texts.append(paper.title + '. ' + paper.summary)
                texts = os.linesep.join(texts)
                word_cloud(texts, str(out_dir / f'images/topic.{i:02d}.texts.png'))

                # report
                wf.write(Report.s(f'## Topic-{i:02d}'))

                wf.write(Report.s('### Analysis of Keywords'))
                wf.write(Report.s('#### Word Cloud'))
                wf.write(Report.s(f'<img src="images/topic.{i:02d}.keywords.png">'))
                wf.write(Report.s('#### Word Frequency (most_common(5))'))
                for keyword, freq in topics[i]['keyword_freqdist'].most_common(5):
                    wf.write(Report.l(f'{keyword} ({freq})'))

                wf.write(Report.s('### Analysis of Title & Summary'))
                wf.write(Report.s('#### Word Cloud'))
                wf.write(Report.s(f'<img src="images/topic.{i:02d}.texts.png">'))
                wf.write(Report.s('#### Word Frequency (most_common(5))'))
                for keyword, freq in topics[i]['text_freqdist'].most_common(5):
                    wf.write(Report.l(f'{keyword} ({freq})'))
