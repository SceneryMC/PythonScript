import os
import time
from pixivpy3 import *
from pathvalidate import sanitize_filename
from path_cross_platform import path_fit_platform
from secret import pd_path, pd_user_list, pd_token, proxies, pd_pid

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


def get_file_pids(raw_data):
    return set(work.id for work in raw_data)


def download_with_retry(file, path):
    while True:
        try:
            api.download(file, path=path)
            break
        except:
            print('TOO FAST!')
            time.sleep(5)


def get_marked(method, _id, root_dir=None, inc_download=True):
    def func_with_retry(f, *args, **kwargs):
        while True:
            try:
                result = f(*args, **kwargs)
                break
            except:
                print('GET LIST TOO FAST!')
                time.sleep(5)
        return result

    func = getattr(api, method)
    json_result = func_with_retry(func, _id)
    cur_path = get_root_path(root_dir)
    downloaded_files = set(int(file.split('_')[0]) for files in (x for _, _, x in os.walk(cur_path)) for file in files)
    while json_result.illusts is not None:
        download_list(json_result.illusts, cur_path)
        if json_result.next_url is None or \
                (inc_download and len(downloaded_files & get_file_pids(json_result.illusts)) != 0):
            break
        next = api.parse_qs(json_result.next_url)
        json_result = func_with_retry(func, **next)


def download_list(ls, cur_path):
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


if __name__ == '__main__':
    method = input("抓取：")
    inc = input('增量？') != 'False'
    if method == 'ul':
        for user_id, user_name in user_list:
            print(f'----------------{user_name}----------------')
            get_marked('user_illusts', user_id, user_name, inc)
    elif method == 'b':
        get_marked('user_bookmarks_illust', pd_pid, '!BOOKMARK', inc)
