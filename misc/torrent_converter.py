import os
import re
import shutil
import bencode
from path_cross_platform import path_fit_platform
from secret import misc_linux_fastresume_path, misc_windows_fastresume_path

linux_fastresume_path: str = misc_linux_fastresume_path
windows_fastresume_path: str = path_fit_platform(misc_windows_fastresume_path)
linux_fastresume = set(os.listdir(linux_fastresume_path))
windows_fastresume = set(os.listdir(windows_fastresume_path))
windows_output_path = windows_fastresume_path
linux_output_path = linux_fastresume_path


def convert_to_Windows(linux_fastresume, windows_fastresume):
    need_process = linux_fastresume - windows_fastresume
    print(need_process)
    for file in (s for s in need_process if s.endswith('.fastresume')):
        d = bencode.bread(os.path.join(linux_fastresume_path, file))
        new_path = path_fit_platform(d['save_path'], dst_platform='Windows')
        d['save_path'] = new_path
        d['qBt-savePath'] = new_path.replace('\\', '/')

        bencode.bwrite(d, os.path.join(windows_output_path, file))
        torrent = file.replace('.fastresume', '.torrent')
        shutil.copy(os.path.join(linux_fastresume_path, torrent), os.path.join(windows_output_path, torrent))


def convert_to_Linux(linux_fastresume, windows_fastresume):
    need_process = windows_fastresume - linux_fastresume
    print(need_process)
    for file in (s for s in need_process if s.endswith('.fastresume')):
        d = bencode.bread(os.path.join(windows_fastresume_path, file))
        new_path = path_fit_platform(d['save_path'], dst_platform='Linux')
        d['save_path'] = d['qBt-savePath'] = new_path

        bencode.bwrite(d, os.path.join(linux_output_path, file))
        torrent = file.replace('.fastresume', '.torrent')
        shutil.copy(os.path.join(windows_fastresume_path, torrent), os.path.join(linux_output_path, torrent))


def refresh_moved_fastresume():
    with open('/mnt/E/共享/torrent_hash.txt') as f:
        ls = f.readlines()
    for file in ls:
        file = f"{file[:-1]}.fastresume"
        origin = os.path.join(linux_fastresume_path, file)
        d = bencode.bread(origin)
        new_path = re.sub(r'/mnt/F/存储/视频/电影/时间的沉淀-电影分部', r'/mnt/G/视频/电影/时间的沉淀-电影分部',
                          d['save_path'])
        d['save_path'] = d['qBt-savePath'] = new_path
        bencode.bwrite(d, origin)


if __name__ == '__main__':
    convert_to_Windows(linux_fastresume, windows_fastresume)
    convert_to_Linux(linux_fastresume, windows_fastresume)
    # refresh_moved_fastresume()
