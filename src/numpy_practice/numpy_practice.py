from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import sys

args = sys.argv

img = np.array( Image.open(args[1]) )
print(img.shape)
print(img)
