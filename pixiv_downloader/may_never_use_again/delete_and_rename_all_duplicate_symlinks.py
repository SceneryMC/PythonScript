import json
import os
import re
from collections import defaultdict
from pixiv_downloader.utils import get_rank_folders
from secret import pd_path


def get_pid(file_name):
    r = re.match(r'(\d+)(_\w+)?\.(\w+)', file_name)
    return int(r.group(1))


def get_pids(ls):
    return set(get_pid(file) for file in ls)


def get_all_bookmark_num_symlinks_from_dir():
    def test_duplicate_and_assign(wid, ipath):
        result[ipath].append(wid)

    def list_all_symlinks(path):
        for rank in os.listdir(path):
            if rank in get_rank_folders():
                work_path = os.path.join(path, rank)
                # print(work_path)
                for item in os.listdir(work_path):
                    il = item.split('-')
                    if len(il) > 1 and il[-1].isdigit():
                        item_path = os.path.join(work_path, item)
                        if os.path.isdir(item_path):
                            print(item_path)
                            for work_id in get_pids(os.listdir(item_path)):
                                test_duplicate_and_assign(work_id, item_path)

    result = defaultdict(list)
    root : str = os.path.dirname(pd_path)
    for user in os.listdir(root):
        if user == 'Hood':
            continue
        user_path = os.path.join(root, user)
        list_all_symlinks(user_path)
    p1 : str = os.path.join(os.path.dirname(root), 'pixivpy3-symlink')
    for t in os.listdir(p1):
        p2 : str = os.path.join(p1, t)
        for tag in os.listdir(p2):
            tag_path = os.path.join(p2, tag)
            list_all_symlinks(tag_path)
    return result


def test_duplicate_and_remove():
    with open('local_symlinks.txt', 'r') as f:
        d = json.load(f)
    with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    new_d = defaultdict(list[tuple])
    for k, v in d.items():
        parent = os.path.dirname(k)
        base = os.path.basename(k)
        try:
            s = set(info[str(_id)]['user']['id'] for _id in v)
        except Exception as e:
            print(e)
            s = set()
        user_id = s.pop() if len(s) == 1 else None
        new_d[parent].append((base, user_id))
    for k, v in new_d.items():
        new_v = defaultdict(list)
        for folder, user_id in v:
            new_v[user_id].append(folder)
        new_d[k] = new_v
    relink_list = []
    for path, content in new_d.items():
        for user_id, folders in content.items():
            if user_id is None:
                continue
            for i in range(len(folders)):
                for j in range(i + 1, len(folders)):
                    p1 = os.path.join(path, folders[i])
                    p2 = os.path.join(path, folders[j])
                    if os.path.realpath(p1) == os.path.realpath(p2):
                        relink_list.append((p1, p2))
    return relink_list


def relink():
    with open('results.json', 'r') as f:
        relink_list = json.load(f)
    for p1, p2 in relink_list:
        base = os.path.basename(p1)
        orig = base[:base.rfind('-')]
        ls = []
        for folder in os.listdir(os.path.dirname(p1)):
            sp = folder.split('-')
            if len(sp) == 2 and sp[0] == orig and sp[1].isdigit():
                ls.append(folder)
        print(orig, ls)


if __name__ == "__main__":
    # d = get_all_bookmark_num_symlinks_from_dir()
    # with open('../text_files/local_symlinks.txt', 'w') as f:
    #     json.dump(d, f)
    # ls = test_duplicate_and_remove()
    # with open('../text_files/results.json', 'w') as f:
    #     json.dump(ls, f)
    relink()