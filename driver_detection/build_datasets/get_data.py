import os
import glob
import cv2
import pickle
import math

def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def get_driver_data():
    dr = dict()
    path = os.path.join('../../data/driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

# imgs_rows : row dimension; img_cols : col dimension; color_type : greyscale (if = 1) or 3 colours (if = 3)
def get_train_data(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('../../data/imgs/train/c' + str(j) + '/*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers

def save_train_data(X_train, y_train, driver_id, unique_drivers, color_type=1):
    pickle.dump([X_train, y_train, driver_id, unique_drivers], open('../../data/train_data_color' + str(color_type) + '.pkl','wb'))

def get_test_data(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('../../data/imgs/test/*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

def save_test_data(X_test, X_test_id, color_type=1):
    pickle.dump([X_test, X_test_id], open('../../data/test_data_color' + str(color_type) + '.pkl','wb'))

if __name__ == '__main__':
    # Set parameters
    img_rows = 160; img_cols = 160; color_type = 3;
    # Load train data
    X_train, y_train, driver_id, unique_drivers = get_train_data(img_rows=img_rows, img_cols=img_cols, color_type=color_type)
    # Save train data
    save_train_data(X_train, y_train, driver_id, unique_drivers, color_type=color_type)
    # Load test data
    X_test, X_test_id = get_test_data(img_rows=img_rows, img_cols=img_cols, color_type=color_type)
    # Save test data
    save_test_data(X_test, X_test_id, color_type=color_type)

    


