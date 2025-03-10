from clean_duplicates import is_work_duplicate, get_keypoints_of_a_work, database, database_path, d
from secret import tmp_file_path
from n_image_downloader.utils import last_log, all_log, tmp_keypoints_database, tmp_artist_database, \
    download_list_file, tmp_duplicate_path, generate_test_url, alias
from n_image_downloader_tmp import tmp_get_image, base_url_pre, base_url_suf
from selenium import webdriver
from multiprocessing.dummy import Pool
from collections import defaultdict
import undetected_chromedriver as uc
import re
import os
import shutil
import time
import json
import pickle

test_image_num = 20


def generate_url(work):
    return f"https://nhentai.net/g/{work}"


class NImageDownloader:
    def __init__(self):
        self.d_last = None
        self.d_all = None
        self.driver = None
        self.tmp_keypoints_database = {}
        self.tmp_artist_database = defaultdict(list)
        self.test_url = generate_test_url()

    def init_driver(self):
        self.driver = uc.Chrome(options=webdriver.ChromeOptions())
        self.driver.set_window_size(192, 168)
        self.driver.get(self.test_url)
        time.sleep(30)

    def load_log(self):
        with open(last_log) as f:
            self.d_last = json.load(f)
        with open(all_log) as f:
            self.d_all = json.load(f)
        if self.d_last and (input("last_log未清空！输入clear清空……") == 'clear'):
            self.d_last = {}

    def save_log(self):
        with open(last_log, 'w') as f:
            json.dump(self.d_last, f, ensure_ascii=False, indent=True)
        with open(all_log, 'w') as f:
            json.dump(self.d_all, f, ensure_ascii=False, indent=True)

    def get_basic_info(self, url):
        while True:
            self.driver.get(url)
            src = self.driver.page_source
            if (result := re.search(
                    r'<span class="name">(\d+)</span></a></span></div><div class="tag-container field-name">',
                    src)) is not None:
                n = int(result.group(1))
                break
            if '404 - Not Found' in src:
                return {}, -1, False
            print(src)
            time.sleep(30)

        artist = re.search(r'<span class="tags"><a href="/artist/([^/]+)/"', src)
        parodies = re.search('/parody/([^/]+)/', src)
        return {'artist': alias.get(artist.group(1), artist.group(1)) if artist else "None",
                'tags': re.findall(r'"/tag/([^/]+)/"', src),
                'characters': re.findall(r'"/character/([^/]+)/"', src),
                'parodies': parodies.group(1) if parodies else "None",
                }, n, "/language/chinese/" in src

    def visit_work(self, work, Chinese_only, check_duplication):
        url = generate_url(work)
        d, n, isChinese = self.get_basic_info(url)
        print(f'{url}: n = {n}')

        if n == -1:
            print(f'{url}不存在！')
            return
        if (not Chinese_only or isChinese) and self.download_images(work, n, d['artist'], check_duplication):
            self.d_last[work] = self.d_all[work] = d
            self.tmp_artist_database[d['artist']].append(work)
            self.tmp_keypoints_database[work] = get_keypoints_of_a_work(rf"{tmp_file_path}\{work}")
            self.save_log()
            self.save_tmp_database()
            print(f'{url}完成！')

    def get_internal_info(self, work):
        while True:
            try:
                self.driver.get(f"{generate_url(work)}/1")
                s = self.driver.page_source
                pattern = re.search(
                    r'<img src="https://i(\d)\.nhentai\.net/galleries/(\d+)/\d+\.(jpg|png|gif|webp)',
                    s)
                return pattern.group(1), pattern.group(2)
            except:
                time.sleep(5)
                print("VPN DOWN!")

    def download_images(self, work: str, n, artist, check_duplication):
        path_tmp = rf"{tmp_file_path}\{work}"
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)
        server, inner_serial = self.get_internal_info(work)

        def get_downloaded_images():
            return {int(x.split('.')[0]) for x in os.listdir(path_tmp)}

        def download_a_group(ls):
            while len(rest := ls - get_downloaded_images()) != 0:
                print(rest)
                pool = Pool()
                for i in rest:
                    pool.apply_async(tmp_get_image, args=(i, work, folder, path_tmp,))
                pool.close()
                pool.join()

        folder = f"{base_url_pre}{server}{base_url_suf}/{inner_serial}"
        if check_duplication and len(get_downloaded_images()) < test_image_num:
            download_a_group(set(range(1, min(n, test_image_num) + 1)))
            is_duplicate, p = self.check_duplication(work, path_tmp, artist)
            if is_duplicate == "new" and n <= len(os.listdir(rf"{tmp_file_path}\{p}")) + 2:
                with open(tmp_duplicate_path, 'a', encoding='utf-8') as f:
                    f.write(f"{work}与新作品{p}重复且页数更少，下载终止！\n")
                shutil.rmtree(path_tmp)
                return False
            elif is_duplicate != "unique":
                with open(tmp_duplicate_path, 'a', encoding='utf-8') as f:
                    f.write(f"{work}与作品{p}重复，但刚刚下载或页数更多，下载继续……\n")
            print(f"{work}继续下载!")
        download_a_group(set(range(1, n + 1)))
        return True

    def check_duplication(self, work, path, artist):
        if (p := is_work_duplicate(path, artist, self.tmp_keypoints_database, self.tmp_artist_database, work)) != '':
            return "new", p
        elif (q := is_work_duplicate(path, artist)) != '':
            return "storage", q
        else:
            return "unique", ''

    def save_tmp_database(self):
        with open(tmp_keypoints_database, 'wb') as f:
            pickle.dump(self.tmp_keypoints_database, f)
        with open(tmp_artist_database, 'wb') as f:
            pickle.dump(self.tmp_artist_database, f)

    def merge_databases(self):
        self.tmp_keypoints_database.update(database)
        with open(database_path, 'wb') as f:
            pickle.dump(self.tmp_keypoints_database, f)
        self.tmp_keypoints_database = {}
        self.tmp_artist_database = defaultdict(list)
        self.save_tmp_database()

    def load_tmp_database(self):
        with open(tmp_keypoints_database, 'rb') as f:
            self.tmp_keypoints_database = pickle.load(f)
        with open(tmp_artist_database, 'rb') as f:
            self.tmp_artist_database = pickle.load(f)

    def process_requests(self, allow_redownload, Chinese_only, check_duplication):
        self.init_driver()
        self.load_log()
        self.load_tmp_database()

        with open(download_list_file) as f:
            content: list[str] = re.findall(r"(\d{3,6})", f.read())
        for s in content:
            if s not in self.d_last and (allow_redownload or s not in self.d_all):
                self.visit_work(s, Chinese_only=Chinese_only, check_duplication=check_duplication)

        self.merge_databases()
        self.driver.close()
        self.driver = None


if __name__ == '__main__':
    ar = input('允许重复下载？') == "True"
    co = input('仅下载中文？') != "False"
    cd = input('检测重复作品？') != "False"

    downloader_instance = NImageDownloader()
    downloader_instance.process_requests(allow_redownload=ar, Chinese_only=co, check_duplication=cd)
    # downloader_instance.save_tmp_database()
