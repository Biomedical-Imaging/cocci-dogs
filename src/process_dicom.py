import os
import pydicom
import scipy.misc
from glob import glob

for filename in glob('/baldig/physicstest/ValleyFeverDogs/KnownCocciDogs/*/*/*'):

    if os.path.isfile(filename) and not filename.endswith('.jpg'):
        # try:
        dataset = pydicom.dcmread(filename)

        img = dataset.pixel_array
        print(img.shape)
        scipy.misc.toimage(img).save(filename + '.jpg')

    # SAVE IMAGE
    #