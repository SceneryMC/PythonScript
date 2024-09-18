import json
import os
import shutil
from collections import defaultdict
from pixivpy3.utils import JsonDict
from pixiv_downloader.download_marked import get_ugoira_mp4_filename, path, convert_ugoira_frames, \
    get_ugoira_info, download_and_check_zip
from pixiv_downloader.maintain_symlink import get_all_exist_from_json
from pixiv_downloader.utils import BOOKMARK_ONLY, rank_name, rank, replace_filename
from secret import pd_user_list, pd_path


def split_record():
    with open('../text_files/downloaded_info.json', 'r', encoding='utf-8') as f:
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

    with open('../text_files/downloaded_info.json', 'w', encoding='utf-8') as f:
        json.dump(j_write_back, f, ensure_ascii=False, indent=True)
    with open('../text_files/redownload_ls.json', 'w', encoding='utf-8') as f:
        json.dump(j_ls, f, ensure_ascii=False, indent=True)


def move_wrong_jpg():
    d = get_all_exist_from_json()
    with open('../text_files/redownload_ls.json', 'r', encoding='utf-8') as f:
        j = json.load(f)
    for _id, info in j.items():
        shutil.move(d[_id], './tmp')


def remove_wrong_symlink():
    names = {rank_name(i) for i in range(1, len(rank))}
    for user in os.listdir(os.path.dirname(pd_path)):
        user_root = os.path.join(os.path.dirname(pd_path), user)
        abt_delete = set(os.listdir(user_root)) & names
        for folder in abt_delete:
            print(p := os.path.join(user_root, folder))
            shutil.rmtree(p)


def download_works_in_list(ls, cur_path):
    with open('../text_files/downloaded_info.json', 'r+', encoding='utf-8') as f:
        info = json.load(f)
        count = 0
        for work in ls:
            folder_name = replace_filename(work.title)
            # 以download_with_retry为判断是否下载完成的方法，因为允许BOOKMARK重复下载作品
            if not os.path.exists(os.path.join(cur_path, p := get_ugoira_mp4_filename(work.id))):
                print(cur_path, work.page_count, folder_name)
                ugoira_info = get_ugoira_info(work.id)
                download_and_check_zip(ugoira_info['originalSrc'], cur_path)
                convert_ugoira_frames(cur_path, ugoira_info)
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
    with open('../text_files/redownload_ls.json', 'r', encoding='utf-8') as f:
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
