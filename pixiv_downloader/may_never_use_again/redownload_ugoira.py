import json
import os
import shutil
from collections import defaultdict
from pixivpy3.utils import JsonDict
from pixiv_downloader.download_marked import get_ugoira_mp4_filename, path, convert_ugoira_frames, \
    get_ugoira_info, download_with_retry
from pixiv_downloader.maintain_symlink import get_all_exist_from_json
from pixiv_downloader.utils import BOOKMARK_ONLY, get_folder_name
from secret import pd_user_list, pd_path

dl_database = '../text_files/downloaded_info.json'
redownload_ls = 'redownload_ls.json'


def split_record():
    with open(dl_database, 'r', encoding='utf-8') as f:
        j = json.load(f)

    j_write_back = {}
    j_ls = {}
    for _id, info in j.items():
        if info and info.get('type', None) == 'ugoira':
            work_id = info['filename'].split('_')[0]
            print(work_id)
            assert work_id.isdigit(), info['page_count'] == 1
            info['filename'] = get_ugoira_mp4_filename(work_id)
            j_ls[_id] = info
        else:
            j_write_back[_id] = info
    print(len(j_write_back), len(j_ls))
    assert len(j_write_back) + len(j_ls) == len(j)

    with open(dl_database, 'w', encoding='utf-8') as f:
        json.dump(j_write_back, f, ensure_ascii=False, indent=True)
    with open(redownload_ls, 'w', encoding='utf-8') as f:
        json.dump(j_ls, f, ensure_ascii=False, indent=True)


def move_wrong_jpg():
    d = get_all_exist_from_json(dl_database)
    with open(redownload_ls, 'r', encoding='utf-8') as f:
        j = json.load(f)
    for _id, info in j.items():
        shutil.move(d[_id], './tmp')


def download_works_in_list(ls, cur_path):
    with open(dl_database, 'r+', encoding='utf-8') as f:
        info = json.load(f)
        count = 0
        for work in ls:
            folder_name = get_folder_name(work)
            # 以download_with_retry为判断是否下载完成的方法，因为允许BOOKMARK重复下载作品
            if not os.path.exists(os.path.join(cur_path, p := get_ugoira_mp4_filename(work.id))):
                print(cur_path, work.page_count, folder_name)
                url, frames = get_ugoira_info(work.id)
                download_with_retry(url, cur_path)
                convert_ugoira_frames(cur_path, url, frames)
                work['filename'] = p
            # 将下载信息记入json文件
            if (_id := str(work.id)) not in info:
                info[_id] = work
            else:
                print(f"SKIPPED {_id}")
                count += 1
        if count != len(ls):
            f.seek(0)
            json.dump(info, f, ensure_ascii=False, indent=True)


def redownload():
    def split_list(lst, chunk_size):
        # 使用range的步长，依次取出list的片段
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    m = dict(pd_user_list)
    with open(redownload_ls, 'r', encoding='utf-8') as f:
        j = json.load(f)
    d = defaultdict(list)
    for _id, info in j.items():
        user_id = info['user']['id']
        d[user_id].append(JsonDict(info))
    for user_id, works in d.items():
        for chunk in split_list(works, 10):
            download_works_in_list(chunk,
                                   os.path.join(os.path.dirname(path), m.get(user_id, BOOKMARK_ONLY))
                                   )


if __name__ == '__main__':
    # split_record()
    # move_wrong_jpg()
    # remove_wrong_symlink()
    redownload()
    # result = get_work_info(86446494, False)
    # download_works_in_list([result],
    #                        r'C:\Users\13308\PycharmProjects\SceneryMCPythonScript\pixiv_downloader\may_never_use_again')
