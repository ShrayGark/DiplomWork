import cv2
import numpy as np
import functions as f


def stack_img(data):
    print(len(data))
    result = data[0]['symbol']
    for ex in data[1:]:
        result = np.hstack((result, ex['symbol']))

    cv2.imshow('stack', result)
    cv2.waitKey(0)


def show_contours(img, min_noise_area=0.0002, delete_noise=True):
    # показывает существующие на изображении контуры
    binary = f.img_prep(img)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'contours num = {len(contours)}')
    if delete_noise:
        res = f.del_noise(binary, contours)
    else:
        res = binary.copy()

    s = binary.shape[0] * binary.shape[1]
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if delete_noise:
            flag = not ((w * h) / s < min_noise_area)
        else:
            flag = True
        if hierarchy[0][idx][3] == 0 and flag:
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(f'Show Contours', res)
    cv2.waitKey(0)