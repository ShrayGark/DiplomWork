import functions as f
from pathlib import Path
import cv2

img_path = Path.cwd().joinpath('data').joinpath('diplom_data').joinpath('raw').joinpath('008.jpg')
img = cv2.imread(str(img_path))

f.solve(img)