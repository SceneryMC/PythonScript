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

dl_database = 'text_files/downloaded_info.json'
dl_tmp_new = 'text_files/new_tmp.txt'
dl_database_new = 'text_files/downloaded_info_new.json'
updated_info = 'text_files/updated_info.json'


def rank_name(idx):
    return f"!{rank[idx]}"


def get_rank_idx(n):
    return bisect.bisect_right(rank, n) - 1


def rank_update(orig, curr):
    return get_rank_idx(orig) < get_rank_idx(curr)


def can_uprank(n):
    CONST = 0.8
    return rank[-1] > n >= CONST * rank[-1] or rank[-2] > n >= CONST * rank[-2]


def get_pid(file_name):
    r = re.match(r'(\d+)(_\w+)?\.(\w+)', file_name)
    return int(r.group(1))


def get_pids(ls):
    return set(get_pid(file) for file in ls)


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


def get_folder_name(info):
    return f"{info['id']}-{replace_filename(info['title'])}"


def get_target_name(info) -> str:
    return get_folder_name(info) if info['page_count'] > 1 else info['filename']


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
    for item in os.listdir(root_path):
        if item not in get_rank_folders():
            if os.path.isdir(os.path.join(root_path, item)):
                result.add(item.split('-')[0])
            elif not item.startswith('limit_'):
                result.add(get_pid(item))
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
            if 'OAuth' in result['error']['message']:
                _id = args[0] if args else int(kwargs.get('user_id', -1))
                sys.exit(_id)
                # {'error': {'user_message': '', 'message': 'Error occurred at the OAuth process. Please check your Access Token to fix this. Error Message: invalid_grant', 'reason': '', 'user_message_details': {}}}
            else:
                time.sleep(10)
                # {'error': {'user_message': '', 'message': 'Rate Limit', ...}
    return result


if __name__ == '__main__':
    print(rank_name(get_rank_idx(499)))
    print(rank_name(get_rank_idx(500)))
    print(rank_name(get_rank_idx(501)))