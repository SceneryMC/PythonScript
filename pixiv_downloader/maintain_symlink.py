import json
import os
from pathvalidate import sanitize_filename
from collections import defaultdict
from pixiv_downloader.download_marked import api
from pixiv_downloader.utils import get_downloaded_works, get_pid
from secret import pd_path, pd_user_list, pd_symlink_path, pd_tags

user_id_to_name = dict(pd_user_list)


def get_target_name(info):
    return sanitize_filename(info['title']) if info['page_count'] > 1 else info['filename']


def get_work_info(_id):
    work = api.illust_detail(_id)
    if 'error' in work:
        print(_id, work.error)
        return None
    work = work.illust
    del work['meta_pages'], work['meta_single_page'], work['image_urls']
    return work


def get_all_works_info():
    root = os.path.dirname(pd_path)
    downloaded_pids = get_downloaded_works(root)
    print(len(downloaded_pids))
    with open('text_files/downloaded_info.json', 'r+', encoding='utf-8') as f:
        d = json.load(f)
        missing = downloaded_pids - set(int(n) for n in d.keys())
        print(len(missing))
        for work_id in missing:
            info = get_work_info(work_id)
            print(work_id)
            d[work_id] = info
            f.seek(0)
            json.dump(d, f, ensure_ascii=False, indent=True)


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
        result[int(_id)] = os.path.join(os.path.dirname(pd_path),
                                        user_id_to_name.get(user_id, '!BOOKMARK'),
                                        target_name)
    return result


def map_duplicate_tags_to_one(given_tag):
    given_tag = given_tag.lower()
    for tags, cls in pd_tags:
        if any(given_tag.startswith(tag.lower()) for tag in tags):
            return tags[0], cls
    return None, None


def maintain_symlink():
    def create_new_symlinks(new_tags: set[tuple[str, str]], _id, info):
        for tag, cls in new_tags:
            base = os.path.join(pd_symlink_path, cls, tag, user_id_to_name.get(info['user']['id'], '!BOOKMARK'))
            os.makedirs(base, exist_ok=True)
            if not os.path.exists(p := os.path.join(base, get_target_name(info))):
                os.symlink(downloaded_paths[_id], p)

    downloaded_paths = get_all_exist_from_json()
    with open('text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open('text_files/created_symlinks.json', 'r', encoding='utf-8') as f:
        sym = defaultdict(list, json.load(f))
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        new_tags = {
            result for tag in info['tags']
            if (result := map_duplicate_tags_to_one(tag['name']))[1] is not None
               and result[0] not in sym[_id]
        }
        create_new_symlinks(new_tags, int(_id), info)
        sym[_id].extend(t[0] for t in new_tags)
    with open('text_files/created_symlinks.json', 'w', encoding='utf-8') as f:
        json.dump(dict(sym), f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    maintain_symlink()