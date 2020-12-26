import dlib
import glob
import numpy
import os
import sys

import cv2
import imutils
from skimage import io

from face_rec_service import calc_rec_name, load_face_info

if len(sys.argv) != 2:
    print("缺少要辨識的圖片名稱")
    exit()

# # 需要辨識的人臉圖片名稱
img_path = sys.argv[1]


def main():
    load_face_info()
    # 針對需要辨識的人臉同樣進行處理
    img = io.imread(img_path)
    rec_name, x1, y1 = calc_rec_name(img)

    # 將辨識出的人名印到圖片上面
    cv2.putText(img, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    img = imutils.resize(img, width=600)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Face Recognition", img)
    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
