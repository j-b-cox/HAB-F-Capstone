import numpy as np

dataset_path    = "../LabelData/dataset.npy"

dataset = np.load(dataset_path)

print(type(dataset[0]))