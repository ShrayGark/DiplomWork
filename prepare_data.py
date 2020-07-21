import functions as f
import cv2
from pathlib import Path

img_path = Path.cwd().joinpath('data').joinpath('train data').joinpath('raw').joinpath('3.jpg')

data_path = Path.cwd().joinpath('data').joinpath('train data').joinpath('raw')
save_path = Path.cwd().joinpath('data').joinpath('train data').joinpath('prep')
img = cv2.imread(str(img_path))
f.prepare_dataset(data_path, save_path)