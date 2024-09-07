import bisect
import os
from secret import pd_user_list, pd_path

MAX_STR_LEN = 1000
BOOKMARK_ONLY = '!BOOKMARK'
rank = [0, 250, 500, 1000, 2000, 5000, 10000]


def rank_name(idx):
    return f"!{rank[idx]}"


def get_rank_idx(n):
    return bisect.bisect_right(rank, n) - 1


def get_pid(file_name):
    return int(file_name.split('_')[0])


def get_file_pids(raw_data):
    return set(work.id for work in raw_data)


def print_in_one_line(s):
    s = repr(s)
    if len(s) > MAX_STR_LEN:
        print(f"{s[:MAX_STR_LEN//2]}......{s[-MAX_STR_LEN//2:]}")
    else:
        print(s)


def get_last_downloaded_user():
    downloaded_folders = os.listdir(os.path.dirname(pd_path))
    for _id, name in pd_user_list[::-1]:
        if name in downloaded_folders:
            return name
    return ''


def get_downloaded_works(root_path):
    result = set()
    for _, folders, files in os.walk(root_path):
        if not folders and files:
            result |= set(get_pid(file) for file in files)
        else:
            for file in files:
                result.add(get_pid(file))
    return result
    # return set(int(file.split('_')[0]) for files in (x for _, _, x in os.walk(root_path)) for file in files)