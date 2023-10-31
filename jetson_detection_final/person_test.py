import pyrealsense2 as rs
import cv2
import threading
import numpy as np
import os
from openvino.runtime import Core
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import (
    Detection,
    xywh_to_xyxy,
    xywh_to_tlwh,
    tlwh_to_xyxy,
    compute_color_for_labels,
)

import sys
import traceback

def init():
    global model_path, re_model_path, customer_path, lock, NN_BUDGET, MAX_COSINE_DISTANCE, metric, tracker, detector, extractor, customer_image, threshold_distance, core

    core = Core()
    lock = threading.Lock()
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = current_path + "/models/person-detection-retail-0013.xml"
    re_model_path = current_path + "/models/person-reidentification-retail-0277.xml"
    customer_path = current_path + "/customer/customer.jpg"
    NN_BUDGET = 100
    MAX_COSINE_DISTANCE = 0.5  # threshold of matching object
    metric = NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=70, n_init=3)
    detector = Model(model_path)
    extractor = Model(re_model_path, -1)
    threshold_distance = 3.0
    if os.path.isfile(customer_path):
        desired_height = 256
        desired_width = 128
        customer_image = cv2.imread(customer_path)
        customer_image = cv2.resize(customer_image, (desired_width, desired_height))
        customer_image = customer_image.transpose(2, 0, 1)
        customer_image = np.expand_dims(customer_image, axis=0)


class Model:
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        result = self.compiled_model(input)[self.output_layer]
        return result


def open_realsense_capture():
    # pipeline_1 = rs.pipeline()
    # config_1 = rs.config()
    # config_1.enable_device("920312070850")

    # config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device("918512074284")

    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # pipeline_1.start(config_1)
    pipeline_2.start(config_2)
    return pipeline_2
    # return pipeline_1, pipeline_2


def realsense():
    def realsense_function():
        # pipeline_1,
        pipeline_2 = open_realsense_capture()

        while True:
            # frames_1 = pipeline_1.wait_for_frames()
            # color_frame_1 = frames_1.get_color_frame()
            # depth_frame_1 = frames_1.get_depth_frame()

            frames_2 = pipeline_2.wait_for_frames()
            color_frame_2 = frames_2.get_color_frame()
            depth_frame_2 = frames_2.get_depth_frame()

            # if not frames_1 or not frames_2:
            if not frames_2:
                continue

            # color_image_1 = np.asanyarray(color_frame_1.get_data())
            # frame_1 = color_image_1.copy()

            color_image_2 = np.asanyarray(color_frame_2.get_data())
            frame_2 = color_image_2.copy()

            with lock:
                # if frame_1 is None or frame_2 is None:
                if frame_2 is None:
                    break

                detections_1 = []

                if os.path.isfile(customer_path):
                    box_person(frame_2, depth_frame_2, detections_1)

                # images = np.vstack((frame_1, frame_2))

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", frame_2)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                # pipeline_1.stop()
                pipeline_2.stop()
                break

    realsense_thread = threading.Thread(target=realsense_function)
    realsense_thread.start()
    realsense_thread.join()


def preprocess(frame, height, width):
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image


def batch_preprocess(img_crops, height, width):
    img_batch = np.concatenate(
        [preprocess(img, height, width) for img in img_crops], axis=0
    )
    return img_batch


def process_results(h, w, results, thresh=0.7):
    detections = results.reshape(-1, 7)
    boxes = []
    labels = []
    scores = []
    for i, detection in enumerate(detections):
        _, label, score, xmin, ymin, xmax, ymax = detection
        if score > thresh:
            boxes.append(
                [
                    (xmin + xmax) / 2 * w,
                    (ymin + ymax) / 2 * h,
                    (xmax - xmin) * w,
                    (ymax - ymin) * h,
                ]
            )
            labels.append(int(label))
            scores.append(float(score))

    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])
    return np.array(boxes), np.array(scores), np.array(labels)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def draw_boxes(img, bbox, depth_frame, identities=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        depth_x = int((x1 + x2) / 2)
        depth_y = int((y1 + y2) / 2)
        depth_value = depth_frame.get_distance(depth_x, depth_y)

        if depth_value > 0.1:
            depth_text = f"{depth_value:.2f}m"
            if i == 0:
                id = int(identities[i]) if identities is not None else 0
            else:
                id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            label = "{}{:d}".format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                [0, 255, 0],
                2,
            )
            if depth_value < threshold_distance:
                cv2.putText(
                    img,
                    depth_text + " " + "GO",
                    (x1, y1 + t_size[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    img,
                    depth_text + " " + "STOP",
                    (x1, y1 + t_size[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
    return img


def box_person(frame, depth_frame, detections):
    def box_person_function(frame, depth_frame, detections):
        customer_flag = False
        h, w = frame.shape[:2]
        input_image = preprocess(frame, detector.height, detector.width)
        output = detector.predict(input_image)

        _, f_width = frame.shape[:2]
        bbox_xywh, score, label = process_results(h, w, results=output)

        img_crops = []

        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
            img = frame[y1:y2, x1:x2]
            img_crops.append(img)

        if img_crops:
            # preprocess
            img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
            features = extractor.predict(img_batch)
        else:
            features = np.array([])

        if os.path.isfile(customer_path):
            customer_features = extractor.predict(customer_image)

        bbox_tlwh = xywh_to_tlwh(bbox_xywh)

        for i in range(features.shape[0]):
            sim = cosin_metric(customer_features, features[i])
            if sim >= 1 - MAX_COSINE_DISTANCE:
                print(f"customer({sim})")
                detections = [Detection(bbox_tlwh[i], features[i])]
            else:
                print(f"not customer({sim})")

        tracker.predict()
        tracker.update(detections)

        outputs = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, depth_frame, identities)

    box_person_thread = threading.Thread(
        target=box_person_function, args=(frame, depth_frame, detections)
    )
    box_person_thread.start()
    box_person_thread.join()


if __name__ == "__main__":
    init()
    log_file = open("error_log.txt", "w")
    sys.stderr = log_file
    try:
        realsense()
    except Exception as e:
        # Log the exception and its stack trace
        traceback.print_exc()
    finally:
        log_file.close()
