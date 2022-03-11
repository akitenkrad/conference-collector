import time
import urllib.request

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
