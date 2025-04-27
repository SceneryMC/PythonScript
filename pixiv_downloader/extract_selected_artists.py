import json
import os
import re

from path_cross_platform import path_fit_platform
from pixiv_downloader.maintain_symlink import user_id_to_name

raw_picked = set(os.listdir(r'F:\存储\其它\wallpaper\sp-single')) | set(os.listdir(r'F:\存储\其它\wallpaper\sp-multiple')) | set(os.listdir(r'C:\Users\SceneryMC\Pictures'))
picked = set(s.split('_')[0] for s in raw_picked if re.match(r'\d+_\w+\.\w+', s))
with open('text_files/downloaded_info.json', encoding='utf-8') as f:
    j = json.load(f)
artists = set(j[_id]['user']['id'] for _id in picked if _id in j)
artist_path = [path_fit_platform(os.path.join(r'F:\存储\其它\pixivpy3', user_id_to_name[_id]), dst_platform='Linux') for _id in artists if _id in user_id_to_name]
with open('text_files/results.txt', 'w', encoding='utf-8') as f:
    for path in artist_path:
        f.write(f"+ '{path}'\n")