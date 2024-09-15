import bisect
import json
import os
from typing import Optional
from pathvalidate import sanitize_filename
from collections import defaultdict
from pixiv_downloader.utils import get_pid, rank, rank_name, BOOKMARK_ONLY, get_rank_idx
from secret import pd_path, pd_user_list, pd_symlink_path, pd_tags

user_id_to_name = dict(pd_user_list)


def get_target_name(info) -> str:
    return sanitize_filename(info['title']) if info['page_count'] > 1 else info['filename']


def get_all_exist_from_dir():
    result = {}
    root = os.path.dirname(pd_path)
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for item in os.listdir(user_path):
            if os.path.isdir(item_path := os.path.join(user_path, item)):
                for work_id in set(get_pid(file) for file in os.listdir(item_path)):
                    result[work_id] = item_path
            else:
                result[get_pid(item)] = item_path
    return result


def get_all_exist_from_json():
    result = {}
    with open('text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        user_id = info['user']['id']
        target_name = get_target_name(info)
        user_or_bookmark = BOOKMARK_ONLY if info['is_bookmarked'] else user_id_to_name[user_id]
        result[_id] = os.path.join(os.path.dirname(pd_path),
                                        user_or_bookmark,
                                        target_name)
    return result


def map_duplicate_tags_to_one(given_tag) -> tuple[Optional[str], Optional[str]]:
    given_tag = given_tag.lower()
    for tags, cls in pd_tags:
        if any(given_tag.startswith(tag.lower()) for tag in tags):
            return tags[0], cls
    return None, None


def create_symlinks(work_id, info, sym, downloaded_paths):
    def create_symlink_general(path):
        os.makedirs(path, exist_ok=True)
        if not os.path.exists(p := os.path.join(path, get_target_name(info))):
            os.symlink(downloaded_paths[work_id], p)
    def create_symlink_by_bookmark_num(base_path):
        if idx > 0:
            create_symlink_general(os.path.join(base_path, rank_name(idx)))

    idx = get_rank_idx(info['total_bookmarks'])
    user_id = info['user']['id']
    for tag in info['tags']:
        if (result := map_duplicate_tags_to_one(tag['name']))[1] is None or result[0] in sym[work_id]:
            continue
        tag_projected, cls = result
        base = os.path.join(pd_symlink_path, cls, tag_projected)
        user_path = os.path.join(base, user_id_to_name.get(user_id, BOOKMARK_ONLY))
        create_symlink_general(user_path)
        create_symlink_by_bookmark_num(base)
        sym[work_id].append(tag_projected)
    create_symlink_by_bookmark_num(os.path.dirname(downloaded_paths[work_id]))


def add_new_tags_of_bookmark_num():
    with open('text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        idx = bisect.bisect_right(rank, info['total_bookmarks']) - 1
        if idx > 0:
            d[_id]['tags'].append({'name': rank_name(idx), "translated_name": None})
    with open('text_files/downloaded_info.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def maintain_symlink_template():
    downloaded_paths = get_all_exist_from_json()
    with open('text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open('text_files/created_symlinks.json', 'r', encoding='utf-8') as f:
        sym = defaultdict(list, json.load(f))
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        create_symlinks(_id, info, sym, downloaded_paths)
    with open('text_files/created_symlinks.json', 'w', encoding='utf-8') as f:
        json.dump(dict(sym), f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    maintain_symlink_template()
    # add_new_tags_of_bookmark_num()