# Image Filter test

from PIL import Image, ImageFilter


imgstr = '00007'

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.FIND_EDGES)
image.save(imgstr + 'FE.png') 

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.DETAIL)
image.save(imgstr + 'D.png') 

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.EDGE_ENHANCE)
image.save(imgstr + 'EE.png') 

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
image.save(imgstr + 'EEM.png')

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.EMBOSS)
image.save(imgstr + 'EMb.png')

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.SMOOTH)
image.save(imgstr + 'SM.png')

image = Image.open(imgstr + '.png')
image = image.filter(ImageFilter.SHARPEN)
image.save(imgstr + 'SH.png')