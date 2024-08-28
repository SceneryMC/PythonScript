import os
import shutil
from secret import idohae_hubble

path = idohae_hubble
folders = ["heic", "potw", "opo", "sci", "ann", "others"]
images = os.listdir(path)
for folder in folders:
    images.remove(folder)
for image in images:
    dst_folder = folders[-1]
    for folder in folders[:-1]:
        if image[:len(folder)] == folder:
            dst_folder = folder
            break
    shutil.move(rf"{path}\{image}", rf"{path}\{dst_folder}\{image}")
