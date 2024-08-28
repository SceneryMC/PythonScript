import shutil
from secret import misc_windows_fastresume_path, misc_transplant_dest, misc_hash_path

with open(misc_hash_path, "r") as f:
    torrents = f.readlines()
for torrent in torrents:
    torrent = torrent.strip()
    shutil.copy(rf"{misc_windows_fastresume_path}\{torrent}.torrent", misc_transplant_dest)
    shutil.copy(rf"{misc_windows_fastresume_path}\{torrent}.fastresume", misc_transplant_dest)
