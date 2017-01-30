from chainer.datasets import tuple_datase
from PIL import Image
import glob
import numpy as np

def create_array(path, vector):
    files = glob.glob(path +"/*.png")
    tupple_array = []
    for file in files:
        img = np.array( Image.open(file), dtype=np.float32 )
        tupple_array.append((img, vector))

    return tupple_array
