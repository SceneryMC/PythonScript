import json
import os
import sys
import time
from pixivpy3 import *
from pathvalidate import sanitize_filename
from path_cross_platform import path_fit_platform
from pixiv_downloader.utils import print_in_one_line, get_file_pids, get_downloaded_works
from secret import pd_path, pd_user_list, pd_token, proxies, pd_pid

MAX_PAGE = 25
WORKS_PER_PAGE = 30
path = path_fit_platform(pd_path)
user_list: list[tuple[int, str]] = pd_user_list  # [(uid, user_name), ...]

api = AppPixivAPI(proxies=proxies)
api.auth(refresh_token=pd_token)


def get_root_path(root_dir):
    if root_dir is not None:
        cur_path = os.path.join(os.path.dirname(path), root_dir)
        os.makedirs(cur_path, exist_ok=True)
        return cur_path
    return os.path.join(path)


def download_marked(method, _id, root_dir=None, inc_download=True):
    def func_with_retry(f, *args, **kwargs):
        while True:
            try:
                result = f(*args, **kwargs)
                print_in_one_line(result)
                if result.illusts is not None:
                    break
            except:
                print("NETWORK ERROR!")
            else:
                print('FAILED: EMPTY OR ERROR RESULT', args, kwargs)
                sys.exit(1)
                # {'error': {'user_message': '', 'message': 'Error occurred at the OAuth process. Please check your Access Token to fix this. Error Message: invalid_grant', 'reason': '', 'user_message_details': {}}}
        return result

    func = getattr(api, method)
    json_result = func_with_retry(func, _id)
    cur_path = get_root_path(root_dir)
    downloaded_pids = get_downloaded_works(cur_path)
    for _ in range(MAX_PAGE):
        download_works_in_list(json_result.illusts, cur_path)
        if json_result.next_url is None or \
                (inc_download and len(downloaded_pids & get_file_pids(json_result.illusts)) != 0):
            break
        next = api.parse_qs(json_result.next_url)
        json_result = func_with_retry(func, **next)


def download_works_in_list(ls, cur_path):
    def download_with_retry(file, path):
        while True:
            try:
                api.download(file, path=path)
                break
            except:
                print('TOO FAST!')
                time.sleep(5)

    with open('text_files/downloaded_info.json', 'r+', encoding='utf-8') as f:
        info = json.load(f)
        count = 0
        for work in ls:
            folder_name = sanitize_filename(work.title)
            print(cur_path, work.page_count, folder_name)
            if work.page_count > 1:
                tmp_path = os.path.join(cur_path, folder_name)
                os.makedirs(tmp_path, exist_ok=True)
                for meta in work.meta_pages:
                    download_with_retry(meta.image_urls.original, tmp_path)
            else:
                download_with_retry(work.meta_single_page.original_image_url, cur_path)
            if (_id := str(work.id)) not in info:
                del work['meta_pages'], work['meta_single_page'], work['image_urls']
                info[_id] = work
            else:
                print(f"SKIPPED {_id}")
                count += 1
        if count != WORKS_PER_PAGE:
            f.seek(0)
            json.dump(info, f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    method = input("抓取：")
    inc = input('增量？') != 'False'
    if method == 'ul':
        start = input('从哪位作者开始？')
        for user_id, user_name in user_list:
            print(f'----------------{user_name}----------------')
            if user_name == start:
                start = ''
            if start == '':
                download_marked('user_illusts', user_id, user_name, inc)
    elif method == 'b':
        download_marked('user_bookmarks_illust', pd_pid, '!BOOKMARK', inc)
