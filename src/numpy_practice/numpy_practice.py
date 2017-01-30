from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import sys

args = sys.argv

img = np.array( Image.open(args[1]), dtype=np.float32 )
print(img.shape)
print((img, 1))
