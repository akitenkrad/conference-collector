import sys
from os import PathLike
from pathlib import Path
import time
import numpy as np
import urllib.request
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

def is_notebook():
    return 'google.colab' in sys.modules or 'ipykernel' in sys.modules

def urlopen(url:str, timeout:float=5.0, retry:int=5, sleep:float=1.0):
    '''open url
    
    Args:
        url (str): url
        timeout (float): wait for "timeout" seconds when the url doesn't return response
        retry (int): retry at most "retry" times
        sleep (float): sleep after each url call

    Returns:
        return of urllib.request.urlopen()
    '''
    retry_count = 0
    while retry_count < retry:
        try:
            response = urllib.request.urlopen(url, timeout=timeout)
            time.sleep(1.0)
            break
        except Exception as ex:
            retry_count += 1
            print(ex)
            print(f'retry -> {retry_count}')
            time.sleep(1.0)
            if retry <= retry_count:
                raise ex
    return response

def word_cloud(input_text:str, out_path:PathLike):
    mask_path = Path(__file__) / '../images/mask.png'
    mask = np.array(Image.open(str(mask_path)).convert('L'), 'f')
    mask = (mask > 128) * 255
    wc = WordCloud(
        font_path=str(Path(__file__) / '../fonts/Utatane-Regular.ttf'),
        background_color='white',
        max_words=200,
        stopwords=set(STOPWORDS),
        contour_width=3,
        contour_color='steelblue',
        mask=mask
    )
    wc.generate(input_text)
    wc.to_file(str(out_path))
