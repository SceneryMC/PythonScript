from PIL import Image

src = r'E:\共享\dst\noelle_birthday\noelle_birthday.png'
img = Image.open(src) # 替换为你的源文件名
for x in range(3):
    for y in range(3):
        # crop的参数为 (左, 上, 右, 下)
        img.crop((x*1000, y*1000, (x+1)*1000, (y+1)*1000)).save(f'{src[:-4]}_{x+1}{y+1}.png')