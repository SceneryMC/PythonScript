import json
import os
import shutil
import subprocess
import sys

from pixiv_downloader.maintain_symlink import user_id_to_name
from pixiv_downloader.utils import get_last_downloaded_user, updated_info, dl_tmp_new
from secret import pd_tags

MAX_ITER = 100


def download_ul():
    inc = input('增量？')
    time_diff_str = input('时限？')
    user = input('从哪位作者开始？')
    for i in range(MAX_ITER):
        print(f'----------------第{i + 1}次循环，从【{user}】继续下载！----------------')
        result = subprocess.run([rf'{os.path.abspath('..')}\venv\Scripts\python.exe', 'download_marked.py'],
                                encoding='utf-8',
                                input=f'ul\n{inc}\n{time_diff_str}\n{user if user != 'AUTO' else get_last_downloaded_user()}\n')
        if result.returncode == 0:
            break
        user = user_id_to_name[result.returncode]


def merge_txt_to_json(path):
    j = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            j.update(json.loads(line))
    with open(path.split('.')[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(j, f, ensure_ascii=False, indent=True)


def get_info_s():
    inc = input('增量？')
    time_diff_str = input('时限？')
    for tags, _type in pd_tags:
        if _type == 'CHARACTER':
            with open(updated_info, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=True)
            for tag in tags:
                print(f'----------------{tags[0]}-{tag}----------------')
                while True:
                    result = subprocess.run([rf'{os.path.abspath('..')}\venv\Scripts\python.exe', 'download_marked.py'],
                                            encoding='utf-8',
                                            input=f"s\n{inc}\n{time_diff_str}\n{tag}\ny\n")
                    if result.returncode == 0:
                        break
            dst = f'text_files/search_results/{tags[0]}.txt'
            shutil.move(dl_tmp_new, dst)
            merge_txt_to_json(dst)


if __name__ == '__main__':
    download_ul()
    # get_info_s()