import json
import os
import time
from pixiv_downloader.download_marked import api, get_downloaded_works
from secret import pd_path


def get_work_info(_id):
    work = api.illust_detail(_id)
    if 'error' in work:
        print(_id, work.error)
        return None
    work = work.illust
    del work['meta_pages'], work['meta_single_page'], work['image_urls']
    return work


def get_all_works_info():
    root = os.path.dirname(pd_path)
    downloaded_pids = get_downloaded_works(root)
    print(len(downloaded_pids))
    with open('text_files/downloaded_info.json', 'r+', encoding='utf-8') as f:
        d = json.load(f)
        missing = downloaded_pids - set(int(n) for n in d.keys())
        for work_id in missing:
            info = get_work_info(work_id)
            d[work_id] = info
            f.seek(0)
            json.dump(d, f, ensure_ascii=False, indent=True)
            time.sleep(1.1)


if __name__ == '__main__':
    get_all_works_info()