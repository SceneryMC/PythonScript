import os
from secret import misc_sd_path

for root, dirs, files in os.walk(misc_sd_path):
    for file in files:
        dst = os.path.join(root, file)
        if os.path.islink(dst):
            src = os.readlink(dst)
            os.remove(dst)
            os.link(src, dst)
            print(src, dst)
