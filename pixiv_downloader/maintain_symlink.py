import bisect
from datetime import datetime, timedelta, timezone
import json
import os
import shutil
from typing import Optional
from pixiv_downloader.utils import get_pid, rank, rank_name, BOOKMARK_ONLY, get_rank_idx, get_target_name, \
    get_rank_folders, dl_database, updated_info, get_pids, can_uprank
from secret import pd_path, pd_user_list, pd_symlink_path, pd_tags

user_id_to_name = dict(pd_user_list)


def remove_wrong_symlink():
    names = {rank_name(i) for i in range(1, len(rank))}
    for user in os.listdir(os.path.dirname(pd_path)):
        user_root = os.path.join(os.path.dirname(pd_path), user)
        abt_delete = set(os.listdir(user_root)) & names
        for folder in abt_delete:
            print(p := os.path.join(user_root, folder))
            shutil.rmtree(p)


def get_all_exist_from_dir():
    def test_duplicate_and_assign(wid, ipath):
        if wid in result:
            print(f"DUPLICATE:{wid} exists at {result[wid]} and {ipath}")
        else:
            result[wid] = ipath
    result = {}
    root = os.path.dirname(pd_path)
    for user in os.listdir(root):
        if user == 'Hood':
            continue
        user_path = os.path.join(root, user)
        for item in os.listdir(user_path):
            if item not in get_rank_folders() and item not in {'!UGOIRA', 'limit_mypixiv_360.png', 'limit_sanity_level_360.png', 'limit_unknown_360.png'}:
                if os.path.isdir(item_path := os.path.join(user_path, item)):
                    for work_id in get_pids(os.listdir(item_path)):
                        test_duplicate_and_assign(work_id, item_path)
                else:
                    test_duplicate_and_assign(get_pid(item), item_path)
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


def map_duplicate_tags_to_one(given_tag, target_tags=pd_tags) -> tuple[Optional[str], Optional[str]]:
    given_tag = given_tag.lower()
    for tags, cls in target_tags:
        if any(given_tag.startswith(tag.lower()) for tag in tags):
            return tags[0], cls
    return None, None


def create_symlinks(work_id, info, downloaded_paths, updated):
    def remove_old_num_symlink(path):
        if os.path.islink(p := os.path.join(path, dst_name)):
            os.remove(p)
            print(f"DELETED SYMLINK: {p}")
        else:
            print(f"DELETE SYMLINK ERROR: {p}")

    def create_symlink_general(path):
        os.makedirs(path, exist_ok=True)
        if not os.path.islink(p := os.path.join(path, dst_name)):
            os.symlink(downloaded_paths[work_id], p)
            print(f'CREATED: {downloaded_paths[work_id]} to {p}')

    def maintain_symlink_by_bookmark_num_and_type(base_path):
        if idx > 0:
            create_symlink_general(os.path.join(base_path, rank_name(idx)))
        if (old_idx := get_rank_idx(updated.get(work_id, 0))) > 0:
            remove_old_num_symlink(os.path.join(base_path, rank_name(old_idx)))
        if dst_name.endswith('.mp4'):
            create_symlink_general(os.path.join(base_path, '!UGOIRA'))

    idx = get_rank_idx(info['total_bookmarks'])
    user_id = info['user']['id']
    results = set(map_duplicate_tags_to_one(tag['name']) for tag in info['tags'])
    dst_name = get_target_name(info)
    for tag_projected, cls in results:
        if cls is not None:
            base = os.path.join(pd_symlink_path, cls, tag_projected)
            user_path = os.path.join(base, user_id_to_name.get(user_id, BOOKMARK_ONLY))
            create_symlink_general(user_path)
            maintain_symlink_by_bookmark_num_and_type(base)
    maintain_symlink_by_bookmark_num_and_type(os.path.dirname(downloaded_paths[work_id]))


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
    with open(os.path.join(os.path.dirname(downloaded_database), 'downloaded_info_new.json'), 'r', encoding='utf-8') as f:
        new_d = json.load(f)
    with open(updated_info, 'r', encoding='utf-8') as f:
        updated_map = {_id: old_num for _id, old_num, new_num in json.load(f) if _id not in new_d and old_num < new_num}
    for _id, info in d.items():
        if info is None or 'user' not in info:
            continue
        create_symlinks(_id, info, downloaded_paths, updated_map)


def merge_updated_bookmark_num(downloaded_database):
    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open(os.path.join(os.path.dirname(downloaded_database), 'downloaded_info_new.json'), 'r', encoding='utf-8') as f:
        new_d = json.load(f)
    assert len(set(d.keys()) & set(new_d.keys())) == 0
    with open(updated_info, 'r', encoding='utf-8') as f:
        updated = json.load(f)

    d.update(new_d)
    for _id, old_num, new_num in updated:
        if _id not in new_d and old_num < new_num:
            d[_id]['total_bookmarks'] = new_num
    with open(downloaded_database, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def uprank_old_close_works(downloaded_database):
    curr = datetime.now(timezone(timedelta(hours=8)))
    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open(updated_info, 'r', encoding='utf-8') as f:
        updated = json.load(f)
    for _id, info in d.items():
        if (not (info is None or 'user' not in info)
                and 'original_total_bookmarks' not in info and can_uprank(info['total_bookmarks'])
                and curr - datetime.fromisoformat(info['create_date']) >= timedelta(seconds=86400 * 180)):
            info['original_total_bookmarks'] = info['total_bookmarks']
            info['total_bookmarks'] = rank[get_rank_idx(info['total_bookmarks']) + 1]
            updated.append([_id, info['original_total_bookmarks'], info['total_bookmarks']])
    with open(downloaded_database, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=True)
    with open(updated_info, 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    # get_all_exist_from_dir()
    # get_all_exist_from_json(dl_database)
    # uprank_old_close_works(dl_database)
    merge_updated_bookmark_num(dl_database)
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
