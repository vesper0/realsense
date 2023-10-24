import pyrealsense2 as rs
import cv2
import numpy as np
import os

from openvino.inference_engine import IECore

ie = IECore()

model_path = os.path.dirname(os.path.abspath(__file__))
model_xml = model_path + "\person-detection-retail-0013.xml"
model_bin = model_path + "\person-detection-retail-0013.bin"

net = ie.read_network(model=model_xml, weights=model_bin)

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

exec_net = ie.load_network(network=net, device_name="CPU")

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

threshold_distance = 3.0

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        input_frame = cv2.resize(color_image, (544, 320))
        input_data = np.expand_dims(input_frame.transpose(2, 0, 1), axis=0)

        results = exec_net.infer(inputs={input_blob: input_data})

        detections = results[output_blob][0][0]

        for detection in detections:
            confidence = detection[2]
            if confidence > 0.8:
                x_min, y_min, x_max, y_max = detection[3:7] * np.array(
                    [
                        color_frame.width,
                        color_frame.height,
                        color_frame.width,
                        color_frame.height,
                    ]
                )
                cv2.rectangle(
                    color_image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2,
                )

                depth_x = int((x_min + x_max) / 2)
                depth_y = int((y_min + y_max) / 2)
                depth_value = depth_frame.get_distance(depth_x, depth_y)
                depth_text = f"{depth_value:.2f}m"

                cv2.putText(
                    color_image,
                    depth_text,
                    (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if depth_value < threshold_distance:
                    cv2.putText(
                        color_image,
                        "GO",
                        (int(x_min), int(y_min) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        color_image,
                        "STOP",
                        (int(x_min), int(y_min) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
