import json
import os
import time
from functools import partial
from pixivpy3 import *
from pathvalidate import sanitize_filename
from path_cross_platform import path_fit_platform
from pixiv_downloader.utils import get_file_pids, get_downloaded_works, BOOKMARK_ONLY, \
    get_info_with_retry
from secret import pd_path, pd_user_list, pd_token, proxies, pd_pid, pd_tags

MAX_PAGE = 1000
FULL_DOWNLOAD_PAGE_LIMIT = 15
path = path_fit_platform(pd_path)
api = AppPixivAPI(proxies=proxies)
api.auth(refresh_token=pd_token)


def criteria_tagged(d, i, tags: set):
    return set(e['name'] for e in d['tags']) & tags if i >= FULL_DOWNLOAD_PAGE_LIMIT else True


def get_root_path(root_dir):
    if root_dir is not None:
        cur_path = os.path.join(os.path.dirname(path), root_dir)
        os.makedirs(cur_path, exist_ok=True)
        return cur_path
    return os.path.join(path)


def download_marked(method, _id, root_dir=None, inc_download=True,
                    criteria=lambda d, i: True):
    func = getattr(api, method)
    json_result = get_info_with_retry(func, _id)
    cur_path = get_root_path(root_dir)
    downloaded_pids = get_downloaded_works(cur_path)
    for i in range(MAX_PAGE):
        ls = [d for d in json_result.illusts if criteria(d, i)]
        download_works_in_list(ls, cur_path)
        if json_result.next_url is None or \
                (inc_download and len(downloaded_pids & get_file_pids(ls)) != 0):
            break
        next = api.parse_qs(json_result.next_url)
        json_result = get_info_with_retry(func, _id, **next)


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
            # 以download_with_retry为判断是否下载完成的方法，因为允许BOOKMARK重复下载作品
            if work.page_count > 1:
                tmp_path = os.path.join(cur_path, folder_name)
                os.makedirs(tmp_path, exist_ok=True)
                for meta in work.meta_pages:
                    download_with_retry(meta.image_urls.original, tmp_path)
            else:
                download_with_retry(work.meta_single_page.original_image_url, cur_path)
                work['filename'] = work.meta_single_page.original_image_url.split('/')[-1]
            if (_id := str(work.id)) not in info:
                del work['meta_pages'], work['meta_single_page'], work['image_urls']
                info[_id] = work
            else:
                print(f"SKIPPED {_id}")
                count += 1
        if count != len(ls):
            f.seek(0)
            json.dump(info, f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    method = input("抓取：")
    inc = input('增量？') != 'False'
    if method == 'ul':
        start = input('从哪位作者开始？')
        for user_id, user_name in pd_user_list:
            if user_name == start:
                start = ''
            if start == '':
                print(f'----------------{user_name}----------------')
                download_marked('user_illusts', user_id, user_name, inc,
                                partial(criteria_tagged, tags=set(elem for tags, cls in pd_tags for elem in tags)))
    elif method == 'b':
        download_marked('user_bookmarks_illust', pd_pid, BOOKMARK_ONLY, inc)
