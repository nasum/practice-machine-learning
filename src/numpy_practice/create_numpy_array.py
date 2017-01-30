from PIL import Image
import glob
import numpy as np
import sys

args = sys.argv

files = glob.glob(args[1] +"/*.png")
tupple_array = []
for file in files:
    img = np.array( Image.open(file), dtype=np.float32 )
    tupple_array.append((img, 1))

print(tupple_array)
