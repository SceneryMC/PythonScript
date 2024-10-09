import bisect
import json
import os
import shutil
from typing import Optional
from pixiv_downloader.utils import get_pid, rank, rank_name, BOOKMARK_ONLY, get_rank_idx, get_target_name
from secret import pd_path, pd_user_list, pd_symlink_path, pd_tags

user_id_to_name = dict(pd_user_list)
dl_database = 'text_files/downloaded_info.json'


def remove_wrong_symlink():
    names = {rank_name(i) for i in range(1, len(rank))}
    for user in os.listdir(os.path.dirname(pd_path)):
        user_root = os.path.join(os.path.dirname(pd_path), user)
        abt_delete = set(os.listdir(user_root)) & names
        for folder in abt_delete:
            print(p := os.path.join(user_root, folder))
            shutil.rmtree(p)


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


def get_all_exist_from_json(downloaded_database):
    result = {}
    with open(downloaded_database, 'r', encoding='utf-8') as f:
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


def create_symlinks(work_id, info, downloaded_paths):
    def create_symlink_general(path):
        os.makedirs(path, exist_ok=True)
        p = os.path.join(path, get_target_name(info))
        if not os.path.islink(p):
            os.symlink(downloaded_paths[work_id], p)
            print(f'CREATED: {downloaded_paths[work_id]} to {p}')
    def create_symlink_by_bookmark_num_and_type(base_path):
        if idx > 0:
            create_symlink_general(os.path.join(base_path, rank_name(idx)))
        if get_target_name(info).endswith('.mp4'):
            create_symlink_general(os.path.join(base_path, '!UGOIRA'))

    idx = get_rank_idx(info['total_bookmarks'])
    user_id = info['user']['id']
    for tag in info['tags']:
        if (result := map_duplicate_tags_to_one(tag['name']))[1] is None:
            continue
        tag_projected, cls = result
        base = os.path.join(pd_symlink_path, cls, tag_projected)
        user_path = os.path.join(base, user_id_to_name.get(user_id, BOOKMARK_ONLY))
        create_symlink_general(user_path)
        create_symlink_by_bookmark_num_and_type(base)
    create_symlink_by_bookmark_num_and_type(os.path.dirname(downloaded_paths[work_id]))


def add_new_tags_of_bookmark_num():
    with open(dl_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        idx = bisect.bisect_right(rank, info['total_bookmarks']) - 1
        if idx > 0:
            d[_id]['tags'].append({'name': rank_name(idx), "translated_name": None})
    with open(dl_database, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def maintain_symlink_template(downloaded_database):
    downloaded_paths = get_all_exist_from_json(downloaded_database)
    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        create_symlinks(_id, info, downloaded_paths)


if __name__ == '__main__':
    maintain_symlink_template(dl_database)
    # add_new_tags_of_bookmark_num()
    # remove_wrong_symlink()

    # with open('text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
    #     j = json.load(f)
    # for _id, info in j.items():
    #     if info and 'user' in info and '_disambiguation' in get_target_name(info):
    #         print(f"{_id:20}{info['user']['name']:30}{info['title']:10}{get_target_name(info):30}")
    #         local_path = os.path.join(os.path.dirname(pd_path), user_id_to_name.get(info['user']['id'], BOOKMARK_ONLY))
    #         files = [s for s in os.listdir(local_path) if s.startswith(str(_id))]
    #         if files:
    #             # assert info['page_count'] > 1
    #             # print("CORRECT", files)
    #             dst = os.path.join(local_path, get_target_name(info))
    #             os.makedirs(dst, exist_ok=True)
    #             for i in range(info['page_count']):
    #                 os.rename(os.path.join(local_path, files[i]), os.path.join(dst, files[i]))
    #         else:
    #             print("MISSING, ONLY TWO")
