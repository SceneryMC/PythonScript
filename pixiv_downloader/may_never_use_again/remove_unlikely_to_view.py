import json
import os
import shutil
from datetime import datetime, timezone, timedelta

from pixiv_downloader.maintain_symlink import get_all_exist_from_json, map_duplicate_tags_to_one, user_id_to_name
from pixiv_downloader.utils import get_rank_idx, get_target_name, BOOKMARK_ONLY, rank_name
from secret import pd_symlink_path, pd_user_list, pd_tags, pd_tmp

time_gap = 86400 * 180 * timedelta(seconds=1)
min_limit_no_tags = 2500
min_limit_with_tags = 750
favorites = {tags[0] for tags, cls in pd_tags[:pd_tags.index((('Firefly', '流萤', '流螢', 'ホタル'), 'CHARACTER'))] for elem in tags}
all_tags = {tags[0] for tags, cls in pd_tags for elem in tags}


def remove_unlikely_to_view(work_id, info, downloaded_paths):
    def remove_symlink_general(path):
        p = os.path.join(path, dst_name)
        print(f'REMOVE: {downloaded_paths[work_id]} to {p}')
        assert os.path.islink(p)
        os.remove(p)

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
    shutil.move(downloaded_paths[work_id], os.path.join(pd_tmp, dst_name))


def main():
    name_id_d = {name: _id for _id, name in pd_user_list}
    with open('../text_files/search_results/size.txt', 'r', encoding='utf-8') as f:
        user_ls = [name_id_d[line.split(":")[0]] for line in f.readlines()[-1000:] if line.split(":")[0] != '!BOOKMARK']
        assert len(user_ls) == 999

    downloaded_paths = get_all_exist_from_json('../text_files/downloaded_info.json')
    curr = datetime.now(timezone(timedelta(hours=8)))
    with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(len(d))
    for _id, info in d.items():
        if info is not None and 'user' in info and info['user']['id'] in user_ls and not info['is_bookmarked'] and os.path.exists(downloaded_paths[_id]):
            tag_of_work = set(e['name'] for e in info['tags'])
            with_tags = len(tag_of_work & all_tags) != 0 and len(tag_of_work & favorites) == 0 and info['total_bookmarks'] < min_limit_with_tags * min((curr - datetime.fromisoformat(info['create_date'])) / time_gap, 1)
            no_tags = len(tag_of_work & all_tags) == 0 and info['total_bookmarks'] < min_limit_no_tags * min((curr - datetime.fromisoformat(info['create_date'])) / time_gap, 1)
            if no_tags or with_tags:
                remove_unlikely_to_view(_id, info, downloaded_paths)



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


def analyze_size():
    with open('../text_files/search_results/size.txt', 'r', encoding='utf-8') as f:
        ls = f.readlines()
    total = sum(int(line.split(':')[1].strip()[:-2]) for line in ls)
    big = sum(int(line.split(':')[1].strip()[:-2]) for line in ls[-1000:])
    print(len(ls), ls[-1000], total, big)


def remove_files_in_tmp():
    all_tmps = [(file.split('-')[0].split('_')[0].split('.')[0], file) for file in os.listdir(pd_tmp)]
    with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    for _id, p in all_tmps:
        if len(set(map_duplicate_tags_to_one(tag['name'])[0] for tag in d[_id]['tags']) & favorites) == 0:
            del d[_id]
            path = os.path.join(pd_tmp, p)
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
            print(os.path.join(pd_tmp, p))
    with open('../text_files/downloaded_info.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def rollback():
    paths = get_all_exist_from_json('../text_files/downloaded_info.json')
    remaining_tmps = [(file.split('-')[0].split('_')[0].split('.')[0], file) for file in os.listdir(pd_tmp)]
    for _id, p in remaining_tmps:
        shutil.move(os.path.join(pd_tmp, p), paths[_id])


if __name__ == '__main__':
    # stat_size()
    # analyze_size()
    main()
    # remove_files_in_tmp()
    # rollback()
