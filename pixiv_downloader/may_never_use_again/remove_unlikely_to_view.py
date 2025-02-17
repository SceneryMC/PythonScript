import json
import os
import shutil
from datetime import datetime, timezone, timedelta

from pixiv_downloader.maintain_symlink import get_all_exist_from_json, map_duplicate_tags_to_one, user_id_to_name
from pixiv_downloader.utils import get_rank_idx, get_target_name, BOOKMARK_ONLY, rank_name
from secret import pd_symlink_path, pd_user_list

time_gap = 86400 * 180 * timedelta(seconds=1)
min_limit = 500


def remove_unlikely_to_view(work_id, info, downloaded_paths):
    def remove_symlink_general(path):
        p = os.path.join(path, dst_name)
        assert os.path.islink(p)
        os.remove(p)
        print(f'REMOVED: {downloaded_paths[work_id]} to {p}')

    def remove_symlink_by_bookmark_num_and_type(base_path):
        if idx > 0:
            remove_symlink_general(os.path.join(base_path, rank_name(idx)))
        if dst_name.endswith('.mp4'):
            remove_symlink_general(os.path.join(base_path, '!UGOIRA'))

    idx = get_rank_idx(info['total_bookmarks'])
    user_id = info['user']['id']
    results = set(map_duplicate_tags_to_one(tag['name']) for tag in info['tags'])
    dst_name = get_target_name(info)
    for tag_projected, cls in results:
        if cls is not None:
            base = os.path.join(pd_symlink_path, cls, tag_projected)
            user_path = os.path.join(base, user_id_to_name.get(user_id, BOOKMARK_ONLY))
            remove_symlink_general(user_path)
            remove_symlink_by_bookmark_num_and_type(base)
    remove_symlink_by_bookmark_num_and_type(os.path.dirname(downloaded_paths[work_id]))
    shutil.move(downloaded_paths[work_id], os.path.join(r'G:\下载\图片\新建文件夹', dst_name))


def main():
    name_id_d = {name: _id for _id, name in pd_user_list}
    with open('../text_files/search_results/size.txt', 'r', encoding='utf-8') as f:
        user_ls = [name_id_d[line.split(":")[0]] for line in f.readlines()[-100:-10]]

    downloaded_paths = get_all_exist_from_json('../text_files/downloaded_info.json')
    curr = datetime.now(timezone(timedelta(hours=8)))
    with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(len(d))
    ls = []
    for _id, info in d.items():
        if info is not None and 'user' in info and info['user']['id'] in user_ls and not info['is_bookmarked']:
            if info['illust_ai_type'] == 2 and info['total_bookmarks'] < min_limit * min((curr - datetime.fromisoformat(info['create_date'])) / time_gap, 1):
                remove_unlikely_to_view(_id, info, downloaded_paths)
    #             ls.append(_id)
    # print(len(ls), ls)


def get_folders_size(path):
    def get_size(folder_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size

    size = round(get_size(path) / 1024 ** 2)
    name = os.path.basename(path)
    print(f"{name}: {size}MB")
    return name, size


def stat_size():
    root = r'F:\存储\其它\pixivpy3'
    ls = []
    for folder in os.listdir(root):
        ls.append(get_folders_size(os.path.join(root, folder)))
    ls.sort(key=lambda x: -x[1])
    print('--------------------------------')
    for name, size in ls:
        print(f"{name}: {size}MB")


if __name__ == '__main__':
    main()
    # with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
    #     d = json.load(f)
    # for file in os.listdir(r'G:\下载\图片\新建文件夹'):
    #     _id = file.split('-')[0].split('_')[0].split('.')[0]
    #     del d[_id]
    # with open('../text_files/downloaded_info.json', 'w', encoding='utf-8') as f:
    #     json.dump(d, f, ensure_ascii=False, indent=True)
