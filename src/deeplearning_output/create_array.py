from chainer.datasets import tuple_dataset
from PIL import Image
import glob
import numpy as np

def create_array(path, vector):
    files = glob.glob(path +"/*.png")

    imageData = []
    labelData = []
    for file in files:
        img = Image.open(file)
        r,g,b = img.split()
        rImgData = np.asarray(np.float32(r)/255.0)
        gImgData = np.asarray(np.float32(g)/255.0)
        bImgData = np.asarray(np.float32(b)/255.0)
        imgData = np.asarray([rImgData, gImgData, bImgData])
        imageData.append(imgData)
        labelData.append(np.int32(vector))

    threshold = np.int32(len(imageData)/8*7)
    train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])

    return train
