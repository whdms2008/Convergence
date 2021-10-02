import random

import PyLidar3
import pandas as pd
import time  # Time module
import cv2


def output(array):
    try:
        df = pd.DataFrame(array)
        df.to_csv('data.csv', index=False, header=False)
    except Exception as e:
        print(e)


# 이미지 형태
# 3등분

def position_out(number, in_frame):
    h, w, c = in_frame.shape
    out_frame = in_frame
    for i in range(1, 3):
        out_frame = cv2.line(out_frame, (int(w / 3 * i), 0), (int(w / 3 * i), h), (255, 255, 255), 1)
    out_frame = cv2.line(out_frame, (0, int(h / 2)), (w, int(h / 2)), (255, 255, 255), 1)
    if number == 1:
        out_frame[0:int(h / 2), 0:int(w / 3)] = 0  # 1번째
    elif number == 2:
        out_frame[0:int(h / 2), int(w / 3 * 1) + 1:int(w / 3 * 2)] = 0  # 2번째
    elif number == 3:
        out_frame[0:int(h / 2), int(w / 3 * 2) + 1:int(w)] = 0  # 3번째
    elif number == 4:
        out_frame[int(h / 2) + 1:h, 0:int(w / 3)] = 0  # 4번째
    elif number == 5:
        out_frame[int(h / 2) + 1:h, int(w / 3 * 1) + 1:int(w / 3 * 2)] = 0  # 5번째
    elif number == 6:
        out_frame[int(h / 2) + 1:h, int(w / 3 * 2) + 1:int(w)] = 0  # 6번째
    return out_frame


def position_lidar(csv):
    data = pd.read_csv(csv , header=None)
    print(data.to_numpy()[0][89:269]) #0.5초 때 값

def videoDetector(vcap):
    # 카메라의 프레임을 지속적으로 받아오기
    while True:
        ret, frame = vcap.read()
        # 이미지 6등분 분할

        frame = cv2.flip(frame, 1)  # 좌우 대칭 변경

        frame = position_out(random.randrange(1,7), frame)

        cv2.imshow("VideoFrame", frame)

        # # cvs2.waitKey(1) 1은 밀리세컨으로 키입력값 대기 지연시간이다. ESC로 멈춤
        if cv2.waitKey(1) == 27:
            vcap.release()  # 메모리 해제
            cv2.destroyAllWindows()  # 모든창 제거, 특정 창만듣을 경우 ("VideoFrame")
            break



position_lidar('data.csv')
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, 960)
cap.set(4, 540)
videoDetector(cap)
while True:
    Obj = PyLidar3.YdLidarX4("COM5")  # PyLidar3s.your_version_of_lidar(port,chunk_size)
    if Obj.Connect():
        print("라이다 연결 됨")
        print("카메라 연결 됨")
        csv_data = []
        csv_out = []
        gen = Obj.StartScanning()
        t = time.time()  # start time
        i = 0
        while (time.time() - t) < 3:  # scan for 30 seconds
            try:
                ret, frame = cap.read()
                position_out(1, frame)
                print("이미지 출력")
                cv2.imwrite('img/data[' + str(i) + '].jpg', frame)
                csv_data.append(list(next(gen).values()))
                time.sleep(0.5)
                i += 1
            except Exception as e:
                print("이미지 처리 오류",e)
                break
        Obj.StopScanning()
        Obj.Disconnect()
        output(csv_data)
        break
    else:
        print("Error connecting to device")
        Obj.Disconnect()
        continue