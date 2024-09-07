import bisect
import json
import os
from pathvalidate import sanitize_filename
from collections import defaultdict
from pixiv_downloader.utils import get_pid, rank, rank_name, BOOKMARK_ONLY
from secret import pd_path, pd_user_list, pd_symlink_path, pd_tags

BOOKMARK_NUM = "BOOKMARK_NUM"
user_id_to_name = dict(pd_user_list)
pd_tags_added = pd_tags + [((rank_name(i), ), BOOKMARK_NUM) for i in range(len(rank))].reverse()


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
        result[_id] = os.path.join(os.path.dirname(pd_path),
                                        user_id_to_name.get(user_id, BOOKMARK_ONLY),
                                        target_name)
    return result


def map_duplicate_tags_to_one(given_tag) -> tuple[str | None, str | None]:
    given_tag = given_tag.lower()
    for tags, cls in pd_tags_added:
        if any(given_tag.startswith(tag.lower()) for tag in tags):
            return tags[0], cls
    return None, None


def create_symlinks(_id, info, sym, downloaded_paths):
    for tag in info['tags']:
        if (result := map_duplicate_tags_to_one(tag['name']))[1] is None or result[0] in sym[_id]:
            continue
        tag_projected, cls = result
        base = os.path.join(pd_symlink_path, cls, tag_projected)
        if cls != BOOKMARK_NUM:
            base = os.path.join(base, user_id_to_name.get(info['user']['id'], BOOKMARK_ONLY))
        os.makedirs(base, exist_ok=True)
        if not os.path.exists(p := os.path.join(base, get_target_name(info))):
            os.symlink(downloaded_paths[_id], p)
        sym[_id].append(tag_projected)


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