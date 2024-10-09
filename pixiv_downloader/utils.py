import bisect
import os
import re
import sys
import time
import zipfile

from pathvalidate import sanitize_filename
from secret import pd_user_list, pd_path

MAX_STR_LEN = 1000
BOOKMARK_ONLY = '!BOOKMARK'
rank = [0, 500, 1000, 2000, 5000, 10000]


def rank_name(idx):
    return f"!{rank[idx]}"


def get_rank_idx(n):
    return bisect.bisect_right(rank, n) - 1


def get_pid(file_name):
    r = re.match(r'(\d+)(_\w+)?\.(\w+)', file_name)
    return int(r.group(1))


def get_file_pids(raw_data):
    return set(work.id for work in raw_data)


def get_ugoira_mp4_filename(_id):
    return f"{_id}.mp4"


def get_name_from_url(url) -> str:
    return url.split("/")[-1]


def get_rank_folders():
    return {rank_name(i) for i in range(1, len(rank))}


def replace_filename(filename):
    new_name = sanitize_filename(filename)
    if new_name in {'..', '.', ''} | get_rank_folders():
        new_name += "_disambiguation"
    return new_name


def get_target_name(info) -> str:
    return replace_filename(info['title']) if info['page_count'] > 1 else info['filename']


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
    for root, folders, files in os.walk(root_path):
        if os.path.basename(root) in get_rank_folders():
            continue
        if not folders and files:
            result |= set(get_pid(file) for file in files)
        else:
            for file in files:
                if not file.startswith('limit_'):
                    result.add(get_pid(file))
    return result
    # return set(int(file.split('_')[0]) for files in (x for _, _, x in os.walk(root_path)) for file in files)


def test_zip(file):
    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            if (p := zip_ref.testzip()) is None:
                return True
    except zipfile.BadZipfile:
        print("BROKEN ZIP!")
    else:
        print("THIS SHOULD NOT HAPPEN!", p)
    return False


def get_info_with_retry(f, keyword='illusts', *args, **kwargs):
    while True:
        try:
            result = f(*args, **kwargs)
            print_in_one_line(result)
            if 'error' not in result and result[keyword] is not None:
                break
        except Exception as e:
            print("NETWORK ERROR!", e)
            time.sleep(5)
        else:
            print('FAILED: EMPTY OR ERROR RESULT', args, kwargs)
            _id = args[0] if args else int(kwargs.get('user_id', -1))
            sys.exit(_id)
            # {'error': {'user_message': '', 'message': 'Error occurred at the OAuth process. Please check your Access Token to fix this. Error Message: invalid_grant', 'reason': '', 'user_message_details': {}}}
    return result
