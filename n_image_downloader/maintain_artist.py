import os
import re
import shutil
from secret import tmp_file_path, artist_path
from n_image_downloader.utils import get_all_exist, info, alias


def get_artist_rank_exist(root_path):
    d = {}
    for base, folder, files in os.walk(root_path):
        if result := re.search(rf'\\ARTIST\\(\d)\\([\w-]+)$', base):
            d[result.group(2)] = int(result.group(1))
    return d


def move_folder_to_artist(from_root_path, to_root_path):
    total = 0
    local_path = get_all_exist(from_root_path)
    local_artist = get_artist_rank_exist(to_root_path)
    for key, value in info.items():
        artist = alias.get(value['artist'], value['artist'])
        rank = local_artist.get(artist, 0)
        if key not in local_path:
            print(f"{key} already moved")
        elif (dst := rf"{to_root_path}\{rank}\{artist}") not in local_path[key]:
            shutil.move(rf"{local_path[key]}\{key}", rf"{dst}\{key}")
            total += 1
            print(rf"{key} from {local_path[key]}\{key} to {dst}\{key}")
        else:
            print(f"UNTOUCHED {key} at {local_path[key]}")
    print(total)


def remove_empty_folder(root_path):
    for base, folder, files in os.walk(root_path):
        if not folder and not files:
            os.removedirs(base)


if __name__ == '__main__':
    move_folder_to_artist(tmp_file_path, artist_path)
    # remove_empty_folder(artist_path)
