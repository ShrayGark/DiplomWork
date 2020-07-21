import cv2
import numpy as np
import os
from pathlib import Path
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import algorithm as alm


def symbol_to_class(symbol):
    d = {'0': 0,
         '1': 1,
         '2': 2,
         '3': 3,
         '4': 4,
         '5': 5,
         '6': 6,
         '7': 7,
         '8': 8,
         '9': 9,
         '+': 10,
         '-': 11}
    return d[symbol]


def class_to_symbol(cls):
    d = {0: '0',
         1: '1',
         2: '2',
         3: '3',
         4: '4',
         5: '5',
         6: '6',
         7: '7',
         8: '8',
         9: '9',
         10: '+',
         11: '-'}
    return d[cls]


def del_noise(frame, contours, min_noise_area=0.0002):
    # удаляет мелкие частицы
    img = frame.copy()
    s = frame.shape[0] * frame.shape[1]
    for contour in contours:
        (x1, y1, w1, h1) = cv2.boundingRect(contour)
        if (w1*h1)/s < min_noise_area:
            matrix = np.array([[255 for j in range(w1)] for i in range(h1)])
            img[y1:y1+h1, x1:x1+w1] = matrix
    return img


def img_prep(img):
    # конвертирует в бинарный вид
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 10)
    return binary


def get_expression_data(img, new_shape=(28, 28), reshape=True, min_noise_area=0.002):
    # возвращает массив из элементов вида [символ, центр, высота, ширина]
    result = []

    binary = img_prep(img)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    binary = del_noise(binary, contours)

    min_s = img.shape[0] * img.shape[1]
    for idx, contour in enumerate(contours):
        metadata = {}
        (x, y, w, h) = cv2.boundingRect(contour)
        metadata['center_x'] = (x + w/2) / binary.shape[0]
        metadata['center_y'] = (y + h/2) / binary.shape[1]
        metadata['height'] = h / binary.shape[1]
        metadata['weight'] = w / binary.shape[0]
        if hierarchy[0][idx][3] == 0 and (w * h) / min_s > min_noise_area:

            thresh_buffer = del_noise(binary.copy(), contour)

            max_side = max(h, w)
            sym_new = 255 * np.ones(shape=[max_side, max_side], dtype=np.uint8)
            if w > h:
                y_pos = max_side // 2 - h // 2
                sym_new[y_pos:y_pos + h, 0:w] = thresh_buffer[y:y + h, x:x + w]
            elif w < h:
                x_pos = max_side // 2 - w // 2
                sym_new[0:h, x_pos:x_pos + w] = thresh_buffer[y:y + h, x:x + w]
            else:
                sym_new = thresh_buffer
            if reshape:
                sym_new = cv2.resize(sym_new, new_shape, interpolation=cv2.INTER_AREA)
            metadata['symbol'] = sym_new
            result.append(metadata)

    return result


def split_img(img, label, delete_noise=True, new_shape=(28, 28), reshape=True, min_noise_area=0.0002):
    result = []

    binary = img_prep(img)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    binary = del_noise(binary, contours)

    min_s = img.shape[0] * img.shape[1]
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if delete_noise:
            flag = ((w * h) / min_s < min_noise_area)
        else:
            flag = True
        if hierarchy[0][idx][3] == 0 and flag:

            thresh_buffer = del_noise(binary.copy(), contour)

            max_side = max(h, w)
            sym_new = 255 * np.ones(shape=[max_side, max_side], dtype=np.uint8)
            if w > h:
                y_pos = max_side // 2 - h // 2
                sym_new[y_pos:y_pos + h, 0:w] = thresh_buffer[y:y + h, x:x + w]
            elif w < h:
                x_pos = max_side // 2 - w // 2
                sym_new[0:h, x_pos:x_pos + w] = thresh_buffer[y:y + h, x:x + w]
            else:
                sym_new = thresh_buffer
            if reshape:
                sym_new = cv2.resize(sym_new, new_shape, interpolation=cv2.INTER_AREA)
            result. append((sym_new, label))
    print(f'split {len(result)}, {label}')
    return result


def prepare_dataset(data_path, save_path):
    result = []
    files = list(os.walk(data_path))[0][2]
    print(files)
    for file in enumerate(files):
        img = cv2.imread(str(data_path.joinpath(file[1])))
        result += split_img(img, symbol_to_class(file[1][:-4]), delete_noise=False)

    np.random.shuffle(result)
    with open(str(save_path.joinpath('data.pickle')), 'wb') as f:
        pickle.dump(result, f)


def load_data():
    data_path = Path.cwd().joinpath('data').joinpath('train data').joinpath('prep').joinpath('data.pickle')

    with open(str(data_path), 'rb') as file:
        data = pickle.load(file)

    train_x = []
    train_y = []
    for example in data[:int(len(data)*0.75)]:
        train_x.append(example[0])
        train_y.append(example[1])

    test_x = []
    test_y = []
    for example in data[int(len(data)*0.75):]:
        test_x.append(example[0])
        test_y.append(example[1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


def data_to_predict():
    img_path = Path.cwd().joinpath('data').joinpath('diplom_data').joinpath('raw').joinpath('02.jpg')
    img = cv2.imread(str(img_path))
    data = get_expression_data(img)
    result = []
    for d in data:
        result.append(d['symbol'])
    return np.array(result)


def fit_model():
    train_x, train_y, test_x, test_y = load_data()

    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)

    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')

    train_x = train_x / 255
    test_x = test_x / 255

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    num_classes = test_y.shape[1]

    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy', 'Recall', 'Precision'])

    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=20, batch_size=100, verbose=2)

    data_path = Path.cwd().joinpath('data').joinpath('model').joinpath('mod')

    model.save(str(data_path))


def get_classified_data(img):
    data = get_expression_data(img)
    symbols = []
    for d in data:
        symbols.append(d['symbol'])
    symbols = np.array(symbols)

    symbols = symbols.reshape(symbols.shape[0], 28, 28, 1).astype('float32')

    data_path = Path.cwd().joinpath('data').joinpath('model').joinpath('mod')
    model = keras.models.load_model(str(data_path))

    res = model.predict_classes(symbols)
    for i, symbol in enumerate(res):
        data[i]['symbol'] = class_to_symbol(symbol)

    # print(res)

    return alm.SymbolKeeper(data)


def solve(img):
    data = get_classified_data(img)
    data.classified_symbols()

    data.count_hv_range()
    data.resolve_uncertainty()

    data.sort_symbols()
    print(f'Символы: {data.symbols}')

    data.level_split()

    data.build_exp_line()

    # data.print_levels()

    data.find_dependence()
    # print()
    data.build_exp()