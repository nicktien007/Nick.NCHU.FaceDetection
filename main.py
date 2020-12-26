import os

import dlib
import cv2
from face_rec_service import load_face_info, calc_rec_name_2
import numpy as np

# 選擇第一隻攝影機
cap = cv2.VideoCapture(0)
# 調整預設影像大小，預設值很大，很吃效能
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# 取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
last_frame_face_cnt = 0
current_frame_face_cnt = 0


def main():
    global last_frame_face_cnt
    global current_frame_face_cnt
    rec_name = ''
    msg = ''

    # 當攝影機打開時，對每個frame進行偵測
    while cap.isOpened():

        key_in = cv2.waitKey(1)
        # 讀出frame資訊
        ret, frame = cap.read()

        # 偵測人臉
        face_rects, scores, idx = detector.run(frame, 0)

        # 上一幀和當前幀中人臉數的計數器，用來避免大量重覆人臉辯識造成FPS下降
        last_frame_face_cnt = current_frame_face_cnt
        current_frame_face_cnt = len(face_rects)

        # 取出偵測的結果
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()

            # 繪製出偵測人臉的矩形範圍
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

            cv2.putText(frame, msg, (x1 + 20, y1 - 10), cv2.FONT_HERSHEY_DUPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, "S: Save current face", (20, 650), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Esc: Quit", (20, 700), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            if key_in == ord('s'):
                register_user(d, frame, x1, x2, y1, y2)

            if (current_frame_face_cnt == last_frame_face_cnt or current_frame_face_cnt == 0) and rec_name != 'Unknown':
                # print(f"   >>> scene 1: 當前幀和上一幀相比沒有發生人臉數變化 / no faces cnt changes in this frame!!!")
                break

            rec_name = calc_rec_name_2(frame)

            msg = 'Unknown???' if rec_name == 'Unknown' else f"{rec_name},Unlock!!"

        # 如果按下ESC鍵，就退出
        if key_in == 27:
            print("exit!!")
            break

        # 輸出到畫面
        cv2.imshow("Face Detection", frame)

    # 釋放記憶體
    cap.release()
    # 關閉所有視窗
    cv2.destroyAllWindows()


def register_user(d, frame, x1, x2, y1, y2):
    existing_faces_cnt = check_existing_faces_cnt()
    # 計算矩形框大小 / Compute the size of rectangle box
    height = (y2 - y1)
    width = (x2 - x1)
    hh = int(height / 2)
    ww = int(width / 2)
    img_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
    for ii in range(height * 2):
        for jj in range(width * 2):
            img_blank[ii][jj] = frame[d.top() - hh + ii][d.left() - ww + jj]
    cv2.imwrite("./rec/" + str("Person_") + str(existing_faces_cnt) + ".jpg", img_blank)
    print("寫入本地 / Save into")
    load_face_info()


# 如果有之前錄入的人臉, 在之前 person_x 的序號按照 person_x+1 開始寫入
def check_existing_faces_cnt():
    if os.listdir("./rec/"):
        # 獲取已錄入的最後一個人臉序號 / Get the order of latest person
        person_list = os.listdir("./rec/")
        return len(person_list) + 1
    # 如果第一次存儲或者沒有之前錄入的人臉, 按照 person_1 開始寫入
    else:
        return 1


if __name__ == '__main__':
    load_face_info()
    main()
