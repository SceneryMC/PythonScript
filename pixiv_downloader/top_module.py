import os
import subprocess

from pixiv_downloader.maintain_symlink import user_id_to_name
from pixiv_downloader.utils import get_last_downloaded_user


MAX_ITER = 100


def download_ul():
    inc = input('增量？')
    user = input('从哪位作者开始？')
    for i in range(MAX_ITER):
        print(f'----------------第{i + 1}次循环，从【{user}】继续下载！----------------')
        result = subprocess.run([rf'{os.path.abspath('..')}\venv\Scripts\python.exe', 'download_marked.py'],
                                encoding='utf-8',
                                input=f'ul\n{inc}\n{user if user != 'AUTO' else get_last_downloaded_user()}\n')
        if result.returncode == 0:
            break
        user = user_id_to_name[result.returncode]


if __name__ == '__main__':
    download_ul()
