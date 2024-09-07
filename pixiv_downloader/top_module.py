import os
import subprocess
from pixiv_downloader.utils import get_last_downloaded_user


MAX_ITER = 100


def download_ul():
    inc = input('增量？')
    user = input('从哪位作者开始？')
    only_tagged = input('是否只下载带有指定标签的作品？')
    for i in range(MAX_ITER):
        print(f'----------------第{i + 1}次循环，从【{user}】继续下载！----------------')
        result = subprocess.run([rf'{os.path.abspath('..')}\venv\Scripts\python.exe', 'download_marked.py'],
                                encoding='utf-8',
                                input=f'ul\n{inc}\n{user if user != 'AUTO' else get_last_downloaded_user()}\n{only_tagged}\n')
        if result.returncode == 0:
            break


if __name__ == '__main__':
    download_ul()
