import json
import os
import re
import time
import undetected_chromedriver as uc
from selenium import webdriver
from collections import defaultdict
from secret import artist_path
from n_image_downloader.utils import alias, all_log, generate_test_url, download_list_file, \
    get_all_works_of_artists

global local_last_work, driver
artist_new_work = 'text_files/n_new_work.json'
works_per_page = 25
js_code = """
ls = []; 
var works = document.querySelectorAll('a.cover');
for (elem of works){
    var url_flag = getComputedStyle(elem.lastChild, ':before')['background-image'];
    ls.push([elem.href, url_flag]);
}
return ls;
"""
flag_CN = 'url("data:image/gif;base64,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")'


def generate_url(artist, page):
    return f"https://nhentai.net/artist/{artist}/?page={page}"


def visit_artist(artist, last_work, Chinese_only):
    page = 1
    works = []
    inf = max(last_work, local_last_work[artist])
    not_over = True
    while not_over:
        tmp_works = []
        while True:
            driver.get(generate_url(artist, page))
            results = driver.execute_script(js_code)
            for tp in results:
                work_id = int(re.match(r'https://nhentai\.net/g/(\d+)/', tp[0]).group(1))
                if work_id > inf and (not Chinese_only or tp[1] == flag_CN):
                    tmp_works.append(work_id)
                elif work_id <= inf:
                    not_over = False
            if not not_over or results or '<h3>No results, sorry.</h3>' in driver.page_source:
                break
            print(driver.page_source)
            time.sleep(30)

        works.extend(tmp_works)
        page += 1
        time.sleep(0.5)
    return works


def visit_artists(last_work, load_func, Chinese_only):
    global local_last_work
    local_last_work = load_func()

    new_works = defaultdict(list)
    for artist in local_last_work.keys():
        new_work = visit_artist(artist, last_work, Chinese_only)
        new_works[alias.get(artist, artist)].extend(new_work)
        print(f"{artist} done! {new_work}")

    with open(artist_new_work, 'w') as f:
        json.dump(new_works, f, ensure_ascii=False, indent=True)


def load_local():
    d = defaultdict(lambda: 0)
    selected = default_artist()
    with open(all_log, encoding='utf-8') as f:
        j = json.load(f)
    for work, info in j.items():
        if 'artist' in info and (artist := info['artist']) in selected and int(work) > d[artist]:
            d[artist] = int(work)
    return d


def default_artist():
    s = set()
    s |= set(os.listdir(rf"{artist_path}\4"))
    s |= set(os.listdir(rf"{artist_path}\5"))
    s |= set(os.listdir(rf"{artist_path}\6"))
    s |= set(alias.keys())

    return s


def load_specified(ignore_existed=True):
    with open(download_list_file) as f:
        d = {s: 0 for s in re.findall(r'https://nhentai\.net/artist/([^/]+)/', f.read())}
    if ignore_existed:
        ignored = set(d.keys()) & set(get_all_works_of_artists().keys())
        for artist in ignored:
            del d[artist]
            print(f"已存在，跳过：{artist}")
    return d


def init_driver():
    global driver
    options = webdriver.ChromeOptions()
    driver = uc.Chrome(options=options)
    driver.set_window_size(192, 168)
    driver.get(generate_test_url())
    time.sleep(30)


if __name__ == '__main__':
    cmd_to_func = {"a": load_local, "s": load_specified}

    co = input('仅下载中文？') != "False"
    last_work = int(input("最近作品？"))
    if last_work == -1:
        with open(all_log) as f:
            m = json.load(f)
            last_work = max(int(x) for x in m.keys())
    target = input("全部a/指定s？")
    print(f"last_work = {last_work}")

    init_driver()
    visit_artists(last_work, cmd_to_func[target], Chinese_only=co)
