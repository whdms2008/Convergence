import PyLidar3
import pandas as pd
import time
import cv2
import threading


def output(array):
    try:
        df = pd.DataFrame(array)
        df.to_csv('data.csv', index=False, header=False)
    except Exception as e:
        print(e)


def position_out(number, in_frame):
    #print(number)
    check = number.keys()
    h, w, c = in_frame.shape
    out_frame = in_frame
    lines = 32
    for j in range(1, lines):
        out_frame = cv2.line(out_frame, (int(w / lines * j), 0), (int(w / lines * j), h), (255, 255, 255), 1)

    out_frame = cv2.line(out_frame, (0, int(h / 2)), (w, int(h / 2)), (255, 255, 255), 1)
    if len(check) == 0:
        print("아무것도 없음 반환")
        return out_frame
    for x in check:
        if x != 1:
            cv2.putText(out_frame, str(int(x)), (int(w/lines * (x-1) ),int(h/4)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            out_frame[0:int(h / 2), int(w / lines * (x-1)) + 1:int(w / lines * x)] = 0  # 2번째
        else:
            out_frame[0:int(h / 2), 0:int(w / lines)] = 0  # 1번째
            cv2.putText(out_frame, str(x), (int(w/lines),int(h/4)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out_frame


def position_lidar(gen):
    numbers = {}
    data = gen[0:89] + gen[269:359]

    j = 5
    for i in range(0, len(data), 5):
        result = sum(data[i:j])
        #print("검출중", i, "도 ~", j, "도 :", result, data[i:j])
        j += 5
        if result <= 1000:
            #print("검출중", i, "도 ~", j, "도 :", result, data[i:j])
            numbers[j/5] = result
    #print(numbers)
    return numbers


class Lidar:
    def __init__(self, port):
        data = PyLidar3.YdLidarX4(port)
        return data


class Logical(threading.Thread):
    def __init__(self):
        super().__init__()
    def run(self):
        global gen
        global lidar
        while True:
            #print("계산 시작")
            lidar = list(next(gen).values())
            #print("계산 완료")


class Camera(threading.Thread):
    def __init__(self, num, width=960, height=540):
        super().__init__()
        self.num = num
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.num, cv2.CAP_DSHOW)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

    def run(self):
        global frame
        global test
        global img
        print("스레드 작동")
        while True:
            ret, frame = self.cap.read()
            if test == 0:
                cv2.imshow("video", frame)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.flip(img,1)
                cv2.imshow("video", img)
            if cv2.waitKey(1) == 27:
                self.cap.release()  # 메모리 해제
                cv2.destroyAllWindows()


if __name__ == '__main__':
    global frame
    global lidar
    global test
    global img
    global gen
    test = 0
    t = Camera(1)
    # sub thread 생성
    t.start()
    b = Logical()

    Obj = PyLidar3.YdLidarX4("COM5")
    while True:  # PyLidar3s.your_version_of_lidar(port,chunk_size)
        if Obj.Connect():
            print("라이다 연결 됨")
            csv_data = []
            csv_out = []
            gen = Obj.StartScanning()
            #print(next(gen))
            b.start()
            t = time.time()  # start time
            i = 0
            while (time.time() - t) < 100:  # scan for 30 seconds
                try:
                    #lidar = list(next(gen).values())
                    # #print("이미지 출력")
                    img = position_out(position_lidar(lidar), frame)
                    test = 1
                    # cv2.imwrite('img/data[' + str(i) + '].jpg', frame)
                    # csv_data.append(lidar)
                    i += 1
                except Exception as e:
                    print("이미지 처리 오류", e)
            Obj.StopScanning()
            Obj.Disconnect()
            output(csv_data)
            break
        else:
            print("라이다 연결 실패")
            Obj.Disconnect()
            continue
