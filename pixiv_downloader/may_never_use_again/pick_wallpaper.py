import json
import shutil

from pixiv_downloader.maintain_symlink import map_duplicate_tags_to_one, get_all_exist_from_json
from pixiv_downloader.utils import replace_filename


def verify(_id, info, downloaded_paths):
    tags = set(tag['name'] for tag in info['tags'])
    if ((info['total_bookmarks'] >= 5000
         or (any(map_duplicate_tags_to_one(tag['name'])[0] in ['Collei', 'Layla'] for tag in info['tags'])
          and info['total_bookmarks'] >= 1000))
            and 'R-18' not in tags
            and any(map_duplicate_tags_to_one(tag['name'])[1] == 'CHARACTER' for tag in info['tags'])
            and info['width'] > info['height'] * 1.1):
        if info['page_count'] == 1:
            shutil.copy(downloaded_paths[_id], r'C:\Users\SceneryMC\Pictures\picked')
        else:
            shutil.copytree(downloaded_paths[_id],
                            rf'C:\Users\SceneryMC\Pictures\picked\{replace_filename(info['title'])}',
                            dirs_exist_ok=True)


def pick_wallpaper(downloaded_database):
    downloaded_paths = get_all_exist_from_json(downloaded_database)
    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        verify(_id, info, downloaded_paths)


if __name__ == '__main__':
    pick_wallpaper('../text_files/downloaded_info.json')