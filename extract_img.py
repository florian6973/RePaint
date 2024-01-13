from PIL import Image
for size in ["s", "m", "l"]:
    for i in range(1, 4):
        path = f'img_{size}_{i}.png'
        img = Image.open(path)
        img_cropped = img.crop((120, 70, 430, 385))
         #img[120:430, 70:385, :]
        img_cropped.save(path + "_cropped.png")

for size in ["s", "m", "l"]:
    for i in range(1, 4):
        path = f'img_{size}_{i}.png'
        img = Image.open(path)
        img_cropped = img.crop((430, 30, 830, 445))
         #img[120:430, 70:385, :]
        img_cropped.save(path + "_cropped_rs.png")