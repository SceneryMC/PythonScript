import json
import os
import re
import shutil

from pixiv_downloader.utils import get_rank_folders, get_target_name


def remove_symlinks():
    root = r'F:\存储\其它\pixivpy3'
    for user in os.listdir(root):
        if user == 'Hood':
            continue
        user_path = os.path.join(root, user)
        for work in os.listdir(user_path):
            if work in get_rank_folders():
                p = os.path.join(user_path, work)
                for link in os.listdir(p):
                    os.remove(os.path.join(p, link))
                os.rmdir(p)
                print(p)


def move_files():
    with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    root = r'F:\存储\其它\pixivpy3'
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for work in os.listdir(user_path):
            p = os.path.join(user_path, work)
            print(p)
            if os.path.isdir(p):
                for file in os.listdir(p):
                    _id, _ = file.split('_')
                    new_folder = os.path.join(user_path, get_target_name(d[_id]))
                    os.makedirs(new_folder, exist_ok=True)
                    # shutil.move(os.path.join(p,file), new_folder)
                    print(os.path.join(p,file), new_folder)


if __name__ == '__main__':
    remove_symlinks()