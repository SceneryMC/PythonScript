import json
import os
import re
import shutil

from pixiv_downloader.maintain_symlink import map_duplicate_tags_to_one, get_all_exist_from_json
from pixiv_downloader.utils import replace_filename
from secret import pd_user_list, pd_wallpaper_dest, pd_processed_max

processed = set(t[0] for t in pd_user_list[:pd_user_list.index((24230399, 'LBZ'))])
pick = {'d': os.listdir(r'C:\Users\SceneryMC\Pictures'),
        'm': set(os.listdir(r'F:\存储\其它\wallpaper\sp-single')) | set(os.listdir(r'F:\存储\其它\wallpaper\sp-multiple'))}


def verify(_picked, _last, _id, info, downloaded_paths, func):
    _tags = set(tag['name'] for tag in info['tags'])

    def desktop():
        return (info['id'] not in _picked
                and not (info['user']['id'] in processed and info['id'] < _last)
                and (info['total_bookmarks'] >= 5000
                     or (any(map_duplicate_tags_to_one(tag['name'])[0] in ['Collei', 'Layla', 'Xiangling'] for tag in
                             info['tags'])
                         and info['total_bookmarks'] >= 1000)
                     )
                and 'R-18' not in _tags
                and any(map_duplicate_tags_to_one(tag['name'])[1] == 'CHARACTER' for tag in info['tags'])
                and info['width'] > info['height'] * 1.1)


    def mobile():
        return (info['id'] not in _picked
                and not (info['user']['id'] in processed and info['id'] < _last)
                and (info['total_bookmarks'] >= 5000
                     or (any(map_duplicate_tags_to_one(tag['name'])[0] in ['Collei', 'Layla', 'Xiangling'] for tag in
                             info['tags'])
                         and info['total_bookmarks'] >= 1000)
                     )
                and 'R-18' not in _tags
                and any(map_duplicate_tags_to_one(tag['name'])[1] == 'CHARACTER' for tag in info['tags'])
                and info['height'] > info['width'] * 1.1)


    func_map = {'d': desktop, 'm': mobile}
    if func_map[func]():
        if info['page_count'] == 1:
            shutil.copy(downloaded_paths[_id], pd_wallpaper_dest)
        else:
            shutil.copytree(downloaded_paths[_id],
                            os.path.join(str(pd_wallpaper_dest), replace_filename(info['title'])),
                            dirs_exist_ok=True)


def pick_wallpaper(downloaded_database):
    downloaded_paths = get_all_exist_from_json(downloaded_database)
    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open(pd_processed_max, 'r', encoding='utf-8') as f:
        last = int(f.read())

    func = input("模式？")
    picked = set(int(s.split('_')[0]) for s in pick[func] if re.match(r'\d+_\w+\.\w+', s))
    for _id, info in d.items():
        if info is not None and 'user' in info:
            verify(picked, last, _id, info, downloaded_paths, func)

    with open(pd_processed_max, 'w', encoding='utf-8') as f:
        f.write(str(max(v['id'] for v in d.values() if v is not None and 'user' in v and v['user']['id'] in processed)))


if __name__ == '__main__':
    pick_wallpaper('text_files/downloaded_info.json')
