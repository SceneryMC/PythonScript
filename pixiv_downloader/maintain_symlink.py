import bisect
from datetime import datetime, timedelta, timezone
import json
import os
import shutil
from pixiv_downloader.utils import get_pid, rank, rank_name, BOOKMARK_ONLY, get_rank_idx, get_target_name, \
    get_rank_folders, dl_database, updated_info, get_pids, can_uprank, map_duplicate_tags_to_one
from secret import pd_path, pd_user_list, pd_symlink_path, pd_tags

user_id_to_name = dict(pd_user_list)
SPECIAL_NAMES = {'!UGOIRA', 'limit_mypixiv_360.png', 'limit_sanity_level_360.png', 'limit_unknown_360.png', 'limit_unviewable_360.png'}


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
        user_path = os.path.join(root, user)
        for item in os.listdir(user_path):
            if item not in get_rank_folders() and item not in SPECIAL_NAMES:
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


def create_links(work_id, info, downloaded_paths, updated, target_tags=pd_tags):
    def remove_old_link(path):
        """
        删除一个旧的链接。
        如果它是一个文件（硬链接），则删除文件。
        如果它是一个目录（包含硬链接的目录），则递归删除整个目录。
        """
        # 构造完整路径
        p = os.path.join(path, dst_name)

        # 使用 lexists 检查路径是否存在，避免在不存在时报错
        if not os.path.lexists(p):
            # 如果路径不存在，可能已经被处理过了，直接返回
            # print(f"路径 '{p}' 不存在，无需删除。")
            return

        try:
            if os.path.isdir(p) and not os.path.islink(p):
                # 如果是一个真实的目录
                shutil.rmtree(p)
                print(f"DELETED DIRECTORY: {p}")
            elif os.path.isfile(p) and not os.path.islink(p):
                # 如果是一个真实的文件（硬链接会被识别为文件）
                os.remove(p)
                print(f"DELETED HARDLINK: {p}")
            # else:
            # 如果是其他情况（例如，意外的软链接），也可以选择处理或忽略
            # print(f"路径 '{p}' 不是预期的文件或目录，跳过删除。")
        except OSError as e:
            print(f"删除 '{p}' 时出错: {e}")

    def create_new_link(path):
        """
        在一个指定路径下，为新文件/文件夹创建链接。
        - 如果是单页作品，创建硬链接。
        - 如果是多页作品，创建一个新目录，并在其中为所有源文件创建硬链接。
        """
        os.makedirs(path, exist_ok=True)
        p = os.path.join(path, dst_name)
        source_path = downloaded_paths[work_id]

        # 核心逻辑：只在目标路径不存在时才进行创建
        if os.path.lexists(p):
            # print(f"路径 '{p}' 已存在，跳过创建。")
            return

        # --- 情况一：单页作品，创建单个硬链接 ---
        if info['page_count'] == 1:
            try:
                os.link(source_path, p)
                print(f'CREATED HARDLINK: {source_path} to {p}')
            except OSError as e:
                print(f"创建硬链接失败: {e}")
        # --- 情况二：多页作品，创建包含硬链接的目录 ---
        elif info['page_count'] > 1:
            try:
                os.makedirs(p)  # 创建目标文件夹
                # 使用 os.scandir() 遍历源文件夹的顶层
                with os.scandir(source_path) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            destination_link = os.path.join(p, entry.name)
                            try:
                                os.link(entry.path, destination_link)
                            except OSError as e_inner:
                                print(f"创建内部硬链接失败: {e_inner}")
                        elif entry.is_dir(follow_symlinks=False):
                            print(f"警告: 在源目录 '{source_path}' 中发现子文件夹 '{entry.path}'，已跳过。")
                print(f"CREATED HARDLINK DIRECTORY: {source_path} to {p}")
            except OSError as e:
                print(f"创建目录 '{p}' 或链接时失败: {e}")

    def maintain_links_by_rank_and_type(base_path):
        """
        根据作品的排名和类型，维护其硬链接或硬链接目录。
        """
        # 1. 为新的排名创建链接
        if idx > 0:
            create_new_link(os.path.join(base_path, rank_name(idx)))
        # 2. 删除旧排名的链接
        if (old_idx := get_rank_idx(updated.get(work_id, 0))) > 0:
            # 确保旧排名和新排名不同，避免不必要的删除和重建
            remove_old_link(os.path.join(base_path, rank_name(old_idx)))
        # 3. 为动图类型创建额外的链接
        if dst_name.endswith('.mp4'):
            create_new_link(os.path.join(base_path, '!UGOIRA'))

    idx = get_rank_idx(info['total_bookmarks'])
    user_id = info['user']['id']
    results = set(map_duplicate_tags_to_one(tag['name'], target_tags=target_tags) for tag in info['tags'])
    dst_name = get_target_name(info)
    for tag_projected, cls in results:
        if cls is not None:
            base = os.path.join(pd_symlink_path, cls, tag_projected)
            user_path = os.path.join(base, user_id_to_name.get(user_id, BOOKMARK_ONLY))
            create_new_link(user_path)
            maintain_links_by_rank_and_type(base)
    maintain_links_by_rank_and_type(os.path.dirname(downloaded_paths[work_id]))


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


def maintain_symlink_template(downloaded_database, target_tags=pd_tags):
    print(target_tags)
    downloaded_paths = get_all_exist_from_json(downloaded_database)

    with open(downloaded_database, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if target_tags is pd_tags:
        with open(os.path.join(os.path.dirname(downloaded_database), 'downloaded_info_new.json'), 'r', encoding='utf-8') as f:
            target_d = json.load(f)
        with open(updated_info, 'r', encoding='utf-8') as f:
            updated_map = {_id: old_num for _id, old_num, new_num in json.load(f) if _id not in target_d and old_num < new_num}
        for _id in updated_map:
            target_d[_id] = d[_id]
    else:
        updated_map = []
        target_d = d

    for _id, info in target_d.items():
        if info is None or 'user' not in info:
            continue
        create_links(_id, info, downloaded_paths, updated_map, target_tags)


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
                and curr - datetime.fromisoformat(info['create_date']) >= timedelta(seconds=86400 * 90)):
            info['original_total_bookmarks'] = info['total_bookmarks']
            info['total_bookmarks'] = rank[get_rank_idx(info['total_bookmarks']) + 1]
            updated.append([_id, info['original_total_bookmarks'], info['total_bookmarks']])
    with open(updated_info, 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    # s1 = set(str(x) for x in get_all_exist_from_dir().keys())
    # s2 = set(get_all_exist_from_json(dl_database).keys())
    # print(s1 - s2)
    # print(s2 - s1)
    # uprank_old_close_works(dl_database)
    merge_updated_bookmark_num(dl_database)
    maintain_symlink_template(dl_database)
    # maintain_symlink_template(dl_database, target_tags=[(('Lanyan', '蓝砚', '藍硯'), 'CHARACTER'),])
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
