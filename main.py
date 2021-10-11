import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import PyLidar3
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './model/yolov4-custom',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'data or data-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.8, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')


def position_pointing(objects):
    obstacle_point = [0, 0, 0]
    distance_point = [0, 0, 0]
    # 장애물 카운트 & 거리 포인트
    for i in range(len(objects)):
        f_x = objects[i][0][0]
        e_x = objects[i][0][-1]
        dist = int(sum(objects[i][1]) / len(objects[i][1]))
        first = f_x // 60
        end = e_x // 60
        cha = abs(first - end)
        if cha == 0:
            obstacle_point[end] += 1
            distance_point[end] += distance(dist)
        else:
            for j in range(cha + 1):
                obstacle_point[first + j] += 1
                distance_point[first + j] += distance(dist)
    return obstacle_point,distance_point


def distance(y):
    if 120 <= y <= 3413:
        return 3
    elif 3414 <= y <= 6706:
        return 2
    elif 6707 <= y <= 10000:
        return 1
    else:
        return 0


def position_detector(lidar):
    data = lidar[270:360] + lidar[0:91]
    # print(len(data))
    if len(data) == 0:
        return None
    object = {}
    cnt = 0
    for i in range(0, len(data)):  # 1
        if data[i - 1] == 0 and data[i] != 0:
            interval = 0
        else:
            interval = abs(data[i] - data[i - 1])  # 현재 distance 값과 라이다의 거리의 차를 계산
        if interval <= 100 and data[i] != 0:  # 값의 차이가 100 이하이면
            cnt += 1  # 카운트 + 1
        if interval > 100 or i == 180 or data[i] == 0:  # 값의 차이가 100 이상이면
            if cnt >= 4:  # 카운트가 5 이상이면
                object[len(object)] = [i - cnt + j for j in range(cnt)], [data[i - cnt + j] for j in range(cnt)]
            cnt = 0  # 카운트 초기화
    return object


def position_lidar(gen, lines):
    numbers = {}
    data = gen[270:359] + gen[0:90]
    if len(data) == 0:
        return numbers
    step = int(len(data) / lines)
    max = step
    for i in range(0, len(data), step):
        if sum(data[i:max]) <= 20000:  # 1000 = 1cm , 10000 = 10cm
            numbers[max / step] = sum(data[i:max])
        max += step;
    return numbers


def main(_argv):
    while True:
        Obj = PyLidar3.YdLidarX4("COM5")
        if Obj.Connect():
            print("라이다 연결됨")
            gen = Obj.StartScanning()
            print("라이다 스캔시작")
            break
        else:
            print("라이다 연결 실패")
            Obj.Disconnect()
            continue
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, _ = utils.load_config(FLAGS)
    input_size = FLAGS.size
    FLAGS.video

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # print(input_details)
        print("모델 불러옴")
        # print(output_details)
        print("모델 실행중")
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(0))
        vid.set(3, 1280)
        vid.set(4, 720)
    except:
        vid = cv2.VideoCapture(0)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        lidar = list(next(gen).values())
        if lidar == 0:
            print("lidar 데이터 없음")
            continue
        lines = 3
        check = position_lidar(lidar, lines)
        return_value, frame = vid.read()
        if return_value:
            h, w, c = frame.shape
            # print(w)
            objects = position_detector(lidar)
            position_pointing(objects)
            for j in range(1, lines):
                frame = cv2.line(frame, (int(w / lines * j), 0), (int(w / lines * j), h), (128, 128, 128), 2)
            cv2.line(frame, (0, int(h / 2)), (w, int(h / 2)), (128, 128, 128), 2)
            cv2.rectangle(frame, (0, h), (w, 0), (128, 128, 128), 4)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # if len(check) != 0:
            #     for x in check:
            #         try:
            #             frame[0:int(h / 2), int(w / lines * (x - 1)) + 1:int(w / lines * x)] = 0  # 2번째
            #         except Exception as e:
            #             print("타입 None")
            #             break
        else:
            print('Video has ended or failed, try a different video format!')
            break

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.8,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
