import h5py
with h5py.File('image.h5', 'r') as dataset:
    class_name = dataset['class_name'][:]
    image = dataset['image'][:]
