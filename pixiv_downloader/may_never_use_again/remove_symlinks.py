import json
import os
import re
import shutil

from pixiv_downloader.utils import get_rank_folders, get_folder_name


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
            possible_id = work.split('-')[0]
            if work != '!UGOIRA' and os.path.isdir(p) and not possible_id in d:
                s = set(file.split('_')[0] for file in os.listdir(p))
                if len(s) == 1:
                    _id = s.pop()
                    if _id in d and d[_id] and 'page_count' in d[_id]:
                        shutil.move(p, os.path.join(os.path.dirname(p), get_folder_name(d[_id])))
                    else:
                        shutil.move(p, os.path.join(r'E:\共享\dst', os.path.basename(p)))
                        print("MISSING!", _id)
                else:
                    for file in os.listdir(p):
                        _id, _ = file.split('_')
                        if _id in d and d[_id] and 'page_count' in d[_id]:
                            src = os.path.join(p, file)
                            new_folder = os.path.join(user_path, get_folder_name(d[_id]))
                            os.makedirs(new_folder, exist_ok=True)
                            shutil.move(src, os.path.join(new_folder, file))
                            # print(os.path.join(p, file), os.path.join(new_folder, file))
                        else:
                            shutil.move(os.path.join(p, file), os.path.join(r'E:\共享\dst', file))
                            print("MISSING!", _id)
                    os.rmdir(p)


if __name__ == '__main__':
    # remove_symlinks()
    move_files()