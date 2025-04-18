import json
import os
import shlex
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone, timedelta
from functools import partial
from pixivpy3 import *
from path_cross_platform import path_fit_platform
from pixiv_downloader.utils import get_file_pids, get_downloaded_works, BOOKMARK_ONLY, \
    get_info_with_retry, get_ugoira_mp4_filename, get_name_from_url, test_zip, rank_update, \
    dl_database, updated_info, dl_database_new, get_folder_name, dl_tmp_new
from secret import pd_path, pd_user_list, pd_token, proxies, pd_pid, pd_tags, pd_headers

MAX_PAGE = 1000
HALF_YEAR = timedelta(days=180)
LEAST_BOOKMARK_NUM = 500
FULL_DOWNLOAD_PAGE_LIMIT = 0
NO_TAG_LIMIT = 4000
FF_CONCAT = '!TMP.txt'
FF_ARGS = '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v baseline -pix_fmt yuv420p -an'

path = path_fit_platform(pd_path)
api = AppPixivAPI(proxies=proxies)
api.auth(refresh_token=pd_token)
start_time = datetime.now(timezone(timedelta(hours=8)))
with open(dl_database, 'r', encoding='utf-8') as f:
    original_info = json.load(f)


def criteria_default(d, i, tags: set):
    half_years = (start_time - datetime.fromisoformat(d['create_date'])) / HALF_YEAR
    intermediate_bool: bool = d['total_bookmarks'] >= LEAST_BOOKMARK_NUM * (1 + int(d['illust_ai_type'] == 2) + 1.5 * int(d["type"] == "ugoira")) * min(half_years, 1)
    return (d['total_bookmarks'] >= NO_TAG_LIMIT or d['is_bookmarked'] or
            (intermediate_bool and (i < FULL_DOWNLOAD_PAGE_LIMIT or set(e['name'] for e in d['tags']) & tags)))


def get_ugoira_info(ugoira_id):
    while True:
        try:
            j = api.ugoira_metadata(int(ugoira_id))
            # result = api.requests_call('GET', f'https://www.pixiv.net/ajax/illust/{ugoira_id}/ugoira_meta',
            #                            headers=auth_header)
            if j and ('error' not in j or not j['error']):
                return j['ugoira_metadata']['zip_urls']['medium'].replace('600x600', '1920x1080'), j['ugoira_metadata']['frames']
        except Exception as e:
            print('NETWORK ERROR:', e)
            time.sleep(5)
        else:
            print('UNEXPECTED ERROR:', ugoira_id, j)
            time.sleep(120)


def convert_ugoira_frames(zip_dirpath: str, ugoira_url, frames, interpolate=False):
    zip_filename = get_name_from_url(ugoira_url)
    zip_file = os.path.join(zip_dirpath, zip_filename)
    work_id = zip_filename.split('_')[0]
    zip_extract_folder = os.path.join(zip_dirpath, work_id)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(zip_extract_folder)
    with open(os.path.join(zip_extract_folder, FF_CONCAT), 'w') as f:
        f.write('ffconcat version 1.0\n\n')
        ugoira_frames = frames.copy()
        last_frame = ugoira_frames[-1].copy()
        last_frame['delay'] = 1
        ugoira_frames.append(last_frame)
        for frame in ugoira_frames:
            frame_file = frame['file']
            frame_duration = frame['delay'] / 1000
            frame_duration = round(frame_duration, 4)
            f.write(
                f'file {frame_file}\n'
                f'duration {frame_duration}\n\n'
            )

    interpolate_arg = '-filter:v "minterpolate=\'fps=60\'"'
    if not interpolate:
        interpolate_arg = ''
    call_str = (
        f'ffmpeg -hide_banner -y '
        f'-i {FF_CONCAT} '
        f'{interpolate_arg} '
        f'{FF_ARGS} '
        f"""'{os.path.join(zip_dirpath, get_ugoira_mp4_filename(work_id))}' """
    )
    call_stack = shlex.split(call_str)
    returncode = subprocess.call(
        call_stack,
        cwd=os.path.abspath(zip_extract_folder),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if returncode == 0:
        shutil.rmtree(zip_extract_folder)
        os.remove(zip_file)
    return returncode


def get_root_path(root_dir):
    if root_dir is not None:
        cur_path = os.path.join(os.path.dirname(path), root_dir)
        os.makedirs(cur_path, exist_ok=True)
        return cur_path
    return os.path.join(path)


def get_work_info(_id, remove_download_link=True):
    work = api.illust_detail(_id)
    if 'error' in work:
        print(_id, work.error)
        return None
    work = work.illust
    if remove_download_link:
        del work['meta_pages'], work['meta_single_page'], work['image_urls']
    return work


def download_with_retry(file_url, path):
    file = os.path.join(path, p := get_name_from_url(file_url))
    while True:
        try:
            api.download(file_url, path=path)
            if not (p.endswith('.zip') and not test_zip(file)):
                break
        except Exception as e:
            print('TOO FAST OR BROKEN FILE!', e)
            time.sleep(5)
        else:
            print('BROKEN ZIP!')
        if os.path.exists(file):
            os.remove(file)


def write_update_info(updated):
    if updated:
        with open(updated_info, 'r', encoding='utf-8') as f:
            ls: list = json.load(f)
        ls.extend(updated)
        with open(updated_info, 'w', encoding='utf-8') as f:
            json.dump(ls, f, ensure_ascii=False, indent=True)


def write_new_work_info(_id, work, updated, new_info, orig_info):
    count_inc = 0
    if _id not in new_info and _id not in orig_info:
        del work['meta_pages'], work['meta_single_page'], work['image_urls']
        new_info[_id] = work
    else:
        count_inc = 1
        bookmark_num = new_info[_id]['total_bookmarks'] if _id in new_info else orig_info[_id][
            'total_bookmarks']
        if rank_update(bookmark_num, work['total_bookmarks']):
            updated.append((_id, bookmark_num, work['total_bookmarks']))
            print(bookmark_num, work['total_bookmarks'])
        else:
            print(f"SKIPPED {_id}", bookmark_num, work['total_bookmarks'])
    return count_inc


def get_work_info_only(ls, cur_path, orig_info):
    updated = []
    with open(dl_tmp_new, 'a', encoding='utf-8') as f:
        new_info = {}
        count = 0
        for work in ls:
            print(f"{work.page_count:3} {work.id} {work.title}")
            _id = str(work.id)
            if work.page_count == 1:  # 此时不可能为ugoira
                if work.type != 'ugoira':
                    work['filename'] = get_name_from_url(work.meta_single_page.original_image_url)
                else:
                    work['filename'] = get_ugoira_mp4_filename(work.id)
            count += write_new_work_info(_id, work, updated, new_info, orig_info) # 将下载信息记入json
        if count != len(ls):
            f.write(f"{json.dumps(new_info)}\n")  # 简单将json序列化后存储，一行一批
    write_update_info(updated)
    time.sleep(1)


def download_works_in_list(ls, cur_path, orig_info):
    updated = []
    with open(dl_database_new, 'r+', encoding='utf-8') as f:
        new_info = json.load(f)
        count = 0
        for work in ls:
            folder_name = get_folder_name(work)
            print(cur_path, work.page_count, folder_name)

            # 以download_with_retry为判断是否下载完成的方法，因为允许BOOKMARK重复下载作品；如果已存在但title不同，视作作者更改了title，跳过
            _id = str(work.id)
            if (_id not in new_info or new_info[_id]['title'] == work.title) and (_id not in orig_info or orig_info[_id]['title'] == work.title):
                if work.page_count > 1:  # 此时不可能为ugoira
                    tmp_path = os.path.join(cur_path, folder_name)
                    os.makedirs(tmp_path, exist_ok=True)
                    for meta in work.meta_pages:
                        download_with_retry(meta.image_urls.original, tmp_path)
                elif work.type != 'ugoira':
                    download_with_retry(work.meta_single_page.original_image_url, cur_path)
                    work['filename'] = get_name_from_url(work.meta_single_page.original_image_url)
                else:
                    work['filename'] = get_ugoira_mp4_filename(work.id)
                    if not os.path.exists(os.path.join(cur_path, work['filename'])):
                        ugoira_url, frames = get_ugoira_info(work.id)
                        download_with_retry(ugoira_url, cur_path)
                        assert convert_ugoira_frames(cur_path, ugoira_url, frames) == 0

            # 将下载信息记入json文件
            count += write_new_work_info(_id, work, updated, new_info, orig_info)
        if count != len(ls):
            f.seek(0)
            json.dump(new_info, f, ensure_ascii=False, indent=True)
    write_update_info(updated)


def download_marked(method, method_kwargs, root_dir=None, inc_download=True, time_diff=86400 * 180,
                    criteria=lambda d, i: True, main_func=download_works_in_list):
    func = getattr(api, method)
    json_result = get_info_with_retry(func, **method_kwargs)
    cur_path = get_root_path(root_dir)
    downloaded_pids = get_downloaded_works(cur_path)
    curr = datetime.now(timezone(timedelta(hours=8)))
    for i in range(MAX_PAGE):
        ls = [d for d in json_result.illusts if criteria(d, i)]
        main_func(ls, cur_path, original_info)
        has_downloaded_files = len(downloaded_pids & get_file_pids(ls)) != 0
        if (json_result.next_url is None
            or (inc_download and has_downloaded_files)
            or (not inc_download and (curr - datetime.fromisoformat(json_result.illusts[-1]['create_date']) >= timedelta(seconds=time_diff)))
        ):
            break
        next = api.parse_qs(json_result.next_url)
        if method == 'search_illust' and int(next['offset']) > 5000:
            next['offset'] = '0'
            next['end_date'] = ls[-1]['create_date'].split('T')[0]
        json_result = get_info_with_retry(func, **next)


def main():
    method = input("抓取：")
    inc = input('增量？') != 'False'
    time_diff = 0x7ffffffff if inc else 86400 * 180
    if time_diff_str := input('时限？'):
        time_diff = eval(time_diff_str)
    if method == 'ul':
        start = input('从哪位作者开始？')
        for user_id, user_name in pd_user_list:
            if user_name == start:
                start = ''
            if start == '':
                print(f'----------------{user_name}----------------')
                download_marked(method='user_illusts',
                                method_kwargs={'user_id': user_id},
                                root_dir=user_name, inc_download=inc, time_diff=time_diff,
                                criteria=partial(criteria_default,
                                                 tags=set(elem for tags, cls in pd_tags for elem in tags)))
    elif method == 'b':
        download_marked(method='user_bookmarks_illust',
                        method_kwargs={'user_id': pd_pid},
                        root_dir=BOOKMARK_ONLY, inc_download=inc, time_diff=time_diff, )
    elif method == 's':
        search_word = input('搜索内容？')
        info_only = input('仅收集信息？') != 'False'
        download_marked(method='search_illust',
                        method_kwargs={'word': search_word},
                        root_dir=os.path.basename(pd_path), inc_download=inc, time_diff=time_diff,
                        main_func=get_work_info_only if info_only else download_works_in_list)
    elif method == 'c':
        pass


if __name__ == '__main__':
    main()