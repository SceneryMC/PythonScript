import json
import os
import re
import shutil
from collections import defaultdict

from pixiv_downloader.maintain_symlink import map_duplicate_tags_to_one, get_all_exist_from_json
from pixiv_downloader.utils import get_target_name
from secret import pd_user_list, pd_wallpaper_dest, pd_processed_max, pd_tags

pick = {'d': os.listdir(r'C:\Users\SceneryMC\Pictures'),
        'm': set(os.listdir(r'F:\存储\其它\wallpaper\sp-single')) | set(os.listdir(r'F:\存储\其它\wallpaper\sp-multiple')),
        'o': [],}


def get_processed(picked, d):
    user_ls = [user_id for user_id, _ in pd_user_list]
    max_item = 0
    for work in picked:
        _id = str(work)
        if _id in d and d[_id]['user']['id'] in user_ls and not d[_id]['is_bookmarked']:
            max_item = max(max_item, user_ls.index(d[_id]['user']['id']))
    return max_item + 1 if picked else 0


def verify(processed, _picked, _last, _id, info, downloaded_paths, func):
    _tags = set(tag for tag in info['tags'])

    def desktop():
        return (info['id'] not in _picked
                and not (info['user']['id'] in processed and info['id'] < _last)
                and (info['total_bookmarks'] >= 5000
                     or (any(map_duplicate_tags_to_one(tag)[0] in ['Collei', 'Layla', 'Xiangling'] for tag in
                             _tags)
                         and info['total_bookmarks'] >= 1000)
                     )
                and 'R-18' not in _tags
                and info["type"] != "ugoira"
                and any(map_duplicate_tags_to_one(tag, target_tags=pd_tags[:28])[0] is not None for tag in _tags)
                and info['width'] > info['height'] * 1.1)


    def mobile():
        return (info['id'] not in _picked
                and not (info['user']['id'] in processed and info['id'] < _last)
                and (info['total_bookmarks'] >= 5000
                     or (any(map_duplicate_tags_to_one(tag)[0] in ['Collei', 'Layla', 'Xiangling'] for tag in
                             _tags)
                         and info['total_bookmarks'] >= 1000)
                     )
                and 'R-18' not in _tags
                and info["type"] != "ugoira"
                and any(map_duplicate_tags_to_one(tag, target_tags=pd_tags[:28])[0] is not None for tag in _tags)
                and info['height'] > info['width'] * 1.1)

    def one_character():
        return (info['total_bookmarks'] >= 5000
                and 'R-18' not in _tags
                and info["type"] != "ugoira"
                and any(map_duplicate_tags_to_one(tag, target_tags=[pd_tags[4]])[0] is not None for tag in _tags)
                )


    func_map = {'d': desktop, 'm': mobile, 'o': one_character}
    if func_map[func]():
        dest = os.path.join(pd_wallpaper_dest, get_target_name(info))
        if not os.path.exists(dest):
            if info['page_count'] == 1:
                shutil.copy(downloaded_paths[_id], dest)
            else:
                shutil.copytree(downloaded_paths[_id], dest)
        else:
            print(f"UNLIKELY: {dest}")


def pick_wallpaper(downloaded_database):
    downloaded_paths = get_all_exist_from_json(downloaded_database)
    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open(pd_processed_max, 'r', encoding='utf-8') as f:
        last = json.load(f)

    func = input("模式？")
    picked = set(int(s.split('_')[0]) for s in pick[func] if re.match(r'\d+_\w+\.\w+', s))
    curr_id = get_processed(picked, d)
    print(curr_id)
    processed = set(_id for _id, _ in pd_user_list[:curr_id])
    for _id, info in d.items():
        if info is not None and 'user' in info:
            verify(processed, picked, last[func], _id, info, downloaded_paths, func)

    with open(pd_processed_max, 'w', encoding='utf-8') as f:
        last[func] = str(max(v['id'] for v in d.values() if v is not None and 'user' in v))
        json.dump(last, f)


if __name__ == '__main__':
    pick_wallpaper('text_files/downloaded_info.json')
