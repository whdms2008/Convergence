import random

import PyLidar3
import numpy as np
import pandas as pd
import time  # Time module
import cv2


def output(array):
    try:
        df = pd.DataFrame(array)
        df.to_csv('data.csv', index=False, header=False)
    except Exception as e:
        print(e)



def position_out(number, in_frame):
    check = number.keys()
    print(check)
    h, w, c = in_frame.shape
    out_frame = in_frame
    for i in range(1, 3):
        out_frame = cv2.line(out_frame, (int(w / 3 * i), 0), (int(w / 3 * i), h), (255, 255, 255), 1)
    out_frame = cv2.line(out_frame, (0, int(h / 2)), (w, int(h / 2)), (255, 255, 255), 1)
    if 1 in check:
        out_frame[0:int(h / 2), 0:int(w / 3)] = 0  # 1번째
    if 2 in check:
        out_frame[0:int(h / 2), int(w / 3 * 1) + 1:int(w / 3 * 2)] = 0  # 2번째
    if 3 in check:
        out_frame[0:int(h / 2), int(w / 3 * 2) + 1:int(w)] = 0  # 3번째
    if 4 in check:
        out_frame[int(h / 2) + 1:h, 0:int(w / 3)] = 0  # 4번째
    if 5 in check:
        out_frame[int(h / 2) + 1:h, int(w / 3 * 1) + 1:int(w / 3 * 2)] = 0  # 5번째
    if 6 in check:
        out_frame[int(h / 2) + 1:h, int(w / 3 * 2) + 1:int(w)] = 0  # 6번째
    return out_frame


def position_lidar(csv):
    numbers = {}
    data = csv[0:89] + csv[269:359]

    j = 5
    for i in range(0, len(data), 5):
        result = sum(data[i:j])
        #print("검출중", i, "도 ~", j, "도 :", result, data[i:j])
        j += 5
        if 500 < result <= 2500:
            if 0 <= i <= 60:
                #print("1번 영역 검출", result)
                numbers[1] = result
                continue
            if 61 <= i <= 120:
                #print("2번 영역 검출", result)
                numbers[2] = result
                continue
            if 121 <= i <= 180:
                #print("3번 영역 검출", result)
                numbers[3] = result
                continue
    return numbers


def videoDetector(vcap):
    # 카메라의 프레임을 지속적으로 받아오기
    while True:
        ret, frame = vcap.read()
        # 이미지 6등분 분할

        frame = cv2.flip(frame, 1)  # 좌우 대칭 변경

        frame = position_out(random.randrange(1, 7), frame)

        cv2.imshow("VideoFrame", frame)

        # # cvs2.waitKey(1) 1은 밀리세컨으로 키입력값 대기 지연시간이다. ESC로 멈춤
        if cv2.waitKey(1) == 27:
            vcap.release()  # 메모리 해제
            cv2.destroyAllWindows()  # 모든창 제거, 특정 창만듣을 경우 ("VideoFrame")
            break


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, 960)
cap.set(4, 540)
# videoDetector(cap)
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
        while (time.time() - t) < 30:  # scan for 30 seconds
            try:
                ret, frame = cap.read()
                lidar = list(next(gen).values())
                position_data = position_lidar(lidar)
                frame = position_out(position_data, frame)
                print("이미지 출력")
                cv2.imwrite('img/data[' + str(i) + '].jpg', frame)
                csv_data.append(lidar)
                time.sleep(0.1)
                i += 1
            except Exception as e:
                print("이미지 처리 오류", e)
                break
            cv2.imshow("VideoFrame", frame)
        Obj.StopScanning()
        Obj.Disconnect()
        output(csv_data)
        break
    else:
        print("라이다 연결 실패")
        Obj.Disconnect()
        continue
