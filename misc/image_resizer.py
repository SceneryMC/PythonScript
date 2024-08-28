import os
from PIL import Image
from path_cross_platform import path_fit_platform
from secret import misc_resizer_base, misc_resizer_dest

base = path_fit_platform(misc_resizer_base)
dest = path_fit_platform(misc_resizer_dest)
ratio = 1 / 2


for image in os.listdir(base):
    img = Image.open(os.path.join(base, image))
    width, height = img.size
    resized = img.resize((int(width * ratio), int(height * ratio)))
    resized.save(os.path.join(dest, image))
    print(f"{image} complete!")
