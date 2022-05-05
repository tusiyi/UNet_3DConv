import os
import cv2


if __name__ == '__main__':
    label_path = "/media/tsy/F/BDD100K/bdd100k_videos_train_00/imgs&labels/label/"
    labels = os.listdir(label_path)

    for label in labels:
        img = cv2.imread(os.path.join(label_path, label), cv2.IMREAD_GRAYSCALE)
        img = img // 2
        cv2.imwrite(os.path.join("./data/label/", label), img)
        print(1)
