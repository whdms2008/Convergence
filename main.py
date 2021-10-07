import threading
import time
import timeit
import PyLidar3
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def output(array):
    try:
        df = pd.DataFrame(array)
        df.to_csv('data.csv', index=False, header=False)
    except Exception as e:
        print(e)


def position_out(number, in_frame):
    # print(number)
    check = number.keys()
    h, w, c = in_frame.shape
    out_frame = in_frame
    lines = 3
    for j in range(1, lines):
        out_frame = cv2.line(out_frame, (int(w / lines * j), 0), (int(w / lines * j), h), (128, 128, 128), 4)

    out_frame = cv2.line(out_frame, (0, int(h / 2)), (w, int(h / 2)), (128, 128, 128), 4)
    out_frame = cv2.line(out_frame, (1, 1), (1, w), (128, 128, 128), 4)
    out_frame = cv2.line(out_frame, (h, h), (1, w), (250, 128, 128), 4)
    out_frame = cv2.line(out_frame, (1, h), (1, 1), (250, 128, 128), 4)
    out_frame = cv2.line(out_frame, (1, h), (w, w), (250, 128, 128), 4)
    if len(check) == 0:
        print("아무것도 없음 반환")
        return None
    for x in check:
            # cv2.putText(out_frame, str(int(x)), (int(w/lines * (x-1) ),int(h/4)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            out_frame[0:int(h / 2), int(w / lines * (x - 1)) + 1:int(w / lines * x)] = 0  # 2번째
            # cv2.putText(out_frame, str(x), (int(w/lines),int(h/4)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return out_frame


def position_lidar(gen):
    numbers = {}
    data = gen[269:359] + gen[0:89]
    #print(len(data))
    j = 5
    if sum(data[0:59]) <= 1500:
        numbers[1] = sum(data[0:59])
    if sum(data[60:119]) <= 1500:
        numbers[2] = sum(data[60:119])
    if sum(data[120:179]) <= 1500:
        numbers[3] = sum(data[120:179])
    # for i in range(0, len(data), 5):
    #     result = sum(data[i:j])
    #     # print("검출중", i, "도 ~", j, "도 :", result, data[i:j])
    #     if result <= 1500:
    #         #print("검출중", i, "도 ~", j, "도 :", result, data[i:j])
    #         #numbers[j/5] = result
    #         if 0 <= i <= 60:
    #             numbers[1] = result
    #         if 61 <= i <= 120:
    #             numbers[2] = result
    #         if 121 <= i <= 180:
    #             numbers[3] = result
    #     j += 5
    #print(numbers)
    return numbers


class Logical(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        global gen
        global lidar
        while True:
            # print("계산 시작")
            start_t = timeit.default_timer()
            lidar = list(next(gen).values())
            #print(len(lidar))
            # print("계산 완료")
            terminate_t = timeit.default_timer()
            #print("걸린시간 ", terminate_t-start_t)


class Camera(threading.Thread):
    def __init__(self, num, width=1920, height=1080):
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
        global lidar
        print("스레드 작동")
        net = cv2.dnn.readNet("yolo/yolov3_custom_2_last.weights", "yolo/yolov3_custom_2.cfg",framework="darknet")
        classes = []
        with open("yolo/obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        while True:
            try:
                ret, frame = self.cap.read()
                img = position_out(position_lidar(lidar), frame)
                if img == None:
                    continue
                else:
                    start_t = timeit.default_timer()
                    # Loading image
                    img = cv2.resize(img, None, fx=0.4, fy=0.4)
                    height, width, channels = img.shape

                    # Detecting objects
                    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # 정보를 화면에 표시
                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                # Object detected
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                # 좌표
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    font = cv2.FONT_HERSHEY_PLAIN
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(img, label, (x, y + 30), font, 1, (255, 0, 0), 3)
                    terminate_t = timeit.default_timer()
                    FPS = int(1./(terminate_t - start_t ))
                    cv2.imshow("Image", img)
                    print(FPS, terminate_t-start_t)
                if cv2.waitKey(1) == 27:
                    self.cap.release()  # 메모리 해제
                    cv2.destroyAllWindows()
            except Exception as e:
                print("라이다 값 없음")


if __name__ == '__main__':
    start_t = timeit.default_timer()
    global frame
    global lidar
    global test
    global gen
    test = 0
    t = Camera(0)
    # sub thread 생성
    b = Logical()
    terminate_t = timeit.default_timer()
    print("걸린시간 ", terminate_t-start_t)
    Obj = PyLidar3.YdLidarX4("COM5")
    while True:  # PyLidar3s.your_version_of_lidar(port,chunk_size)
        if Obj.Connect():
            print("라이다 연결 됨")
            csv_data = []
            csv_out = []
            gen = Obj.StartScanning()
            b.start()
            t.start()
            t = time.time()  # start time
            i = 0
            while (time.time() - t) < 100:  # scan for 30 seconds
                try:
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
