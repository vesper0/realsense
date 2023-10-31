import pyrealsense2 as rs
import cv2
import numpy as np
import os
import threading
import logging as log
import sys
import re

from argparse import ArgumentParser
from pathlib import Path

from openvino.runtime import Core, get_version
from face_utils.util import crop
from face_utils.landmarks_detector import LandmarksDetector
from face_utils.face_detector import FaceDetector
from face_utils.faces_database import FacesDatabase
from face_utils.face_identifier import FaceIdentifier
from face_utils.helpers import resolution
from face_utils.models import OutputTransform


log.basicConfig(
    format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout
)


def open_realsense_capture():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


def init():
    global args, frame_processor, lock, current_path, DEVICE_KINDS

    DEVICE_KINDS = ["CPU", "GPU", "HETERO"]
    lock = threading.Lock()
    current_path = os.path.dirname(os.path.abspath(__file__))
    args = build_argparser().parse_args()
    frame_processor = FrameProcessor(args)


def reload():
    global args, frame_processor
    with lock:
        frame_processor.reload(args)


def realsense():
    def realsense_function():
        pipeline = open_realsense_capture()
        output_transform = None
        input_crop = None
        frame_num = 0
        global frame
        frame = None

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            input_frame = cv2.resize(color_image, (544, 320))
            frame = color_image.copy()

            with lock:
                if frame is None:
                    if frame_num == 0:
                        raise ValueError("Can't read an image from the input")
                    break
                if input_crop is not None:
                    frame = center_crop(frame, input_crop)
                if frame_num == 0:
                    output_transform = OutputTransform(
                        frame.shape[:2], args.output_resolution
                    )
                    if args.output_resolution:
                        output_resolution = output_transform.new_resolution
                    else:
                        output_resolution = (frame.shape[1], frame.shape[0])

                detections = frame_processor.process(frame)
                draw_detections(frame, frame_processor, detections, output_transform)

                images = frame

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", images)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
            elif key == ord("s"):
                save_face(detections, color_image)
            elif key == ord("r"):
                reload()
        pipeline.stop()

    realsense_thread = threading.Thread(target=realsense_function)
    realsense_thread.start()
    realsense_thread.join()


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        default=0,
        help="Required. An input to process. The input must be a single image, "
        "a folder of images, video file or camera id.",
    )
    general.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Optional. Enable reading the input in a loop.",
    )
    general.add_argument(
        "-o",
        "--output",
        help="Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086",
    )
    general.add_argument(
        "-limit",
        "--output_limit",
        default=1000,
        type=int,
        help="Optional. Number of frames to store in output. "
        "If 0 is set, all frames are stored.",
    )
    general.add_argument(
        "--output_resolution",
        default=None,
        type=resolution,
        help="Optional. Specify the maximum output window resolution "
        "in (width x height) format. Example: 1280x720. "
        "Input frame size used by default.",
    )
    general.add_argument(
        "--no_show", action="store_true", help="Optional. Don't show output."
    )
    general.add_argument(
        "--crop_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Crop the input stream to this resolution.",
    )
    general.add_argument(
        "--match_algo",
        default="HUNGARIAN",
        choices=("HUNGARIAN", "MIN_DIST"),
        help="Optional. Algorithm for face matching. Default: HUNGARIAN.",
    )
    general.add_argument(
        "-u",
        "--utilization_monitors",
        default="",
        type=str,
        help="Optional. List of monitors to show initially.",
    )

    gallery = parser.add_argument_group("Faces database")
    gallery.add_argument(
        "-fg",
        default=current_path + "/gall/",
        help="Optional. Path to the face images directory.",
    )
    gallery.add_argument(
        "--run_detector",
        action="store_true",
        help="Optional. Use Face Detection model to find faces "
        "on the face images, otherwise use full images.",
    )
    gallery.add_argument(
        "--allow_grow",
        action="store_true",
        help="Optional. Allow to grow faces gallery and to dump on disk. "
        "Available only if --no_show option is off.",
    )

    models = parser.add_argument_group("Models")
    models.add_argument(
        "-m_fd",
        type=Path,
        default=current_path + "/models/face-detection-retail-0005.xml",
        help="Required. Path to an .xml file with Face Detection model.",
    )
    models.add_argument(
        "-m_lm",
        type=Path,
        default=current_path + "/models/landmarks-regression-retail-0009.xml",
        help="Required. Path to an .xml file with Facial Landmarks Detection model.",
    )
    models.add_argument(
        "-m_reid",
        type=Path,
        default=current_path + "/models/face-reidentification-retail-0095.xml",
        help="Required. Path to an .xml file with Face Reidentification model.",
    )
    models.add_argument(
        "--fd_input_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Specify the input size of detection model for "
        "reshaping. Example: 500 700.",
    )

    infer = parser.add_argument_group("Inference options")
    infer.add_argument(
        "-d_fd",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Detection model. "
        "Default value is CPU.",
    )
    infer.add_argument(
        "-d_lm",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Facial Landmarks Detection "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-d_reid",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Reidentification "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-v", "--verbose", action="store_true", help="Optional. Be more verbose."
    )
    infer.add_argument(
        "-t_fd",
        metavar="[0..1]",
        type=float,
        default=0.6,
        help="Optional. Probability threshold for face detections.",
    )
    infer.add_argument(
        "-t_id",
        metavar="[0..1]",
        type=float,
        default=0.3,
        help="Optional. Cosine distance threshold between two vectors "
        "for face identification.",
    )
    infer.add_argument(
        "-exp_r_fd",
        metavar="NUMBER",
        type=float,
        default=1.15,
        help="Optional. Scaling ratio for bboxes passed to face recognition.",
    )
    return parser


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info("OpenVINO Runtime")
        log.info("\tbuild: {}".format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(
            core,
            args.m_fd,
            args.fd_input_size,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(
            core, args.m_reid, match_threshold=args.t_id, match_algo=args.match_algo
        )

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug("Building faces database using images from {}".format(args.fg))
        self.faces_database = FacesDatabase(
            args.fg,
            self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None,
            args.no_show,
        )
        self.face_identifier.set_faces_database(self.faces_database)
        log.info(
            "Database is built, registered {} identities".format(
                len(self.faces_database)
            )
        )

    def reload(self, args):
        log.debug("Building faces database using images from {}".format(args.fg))
        self.faces_database = FacesDatabase(
            args.fg,
            self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None,
            args.no_show,
        )
        self.face_identifier.set_faces_database(self.faces_database)
        log.info(
            "Database is built, registered {} identities".format(
                len(self.faces_database)
            )
        )

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning(
                "Too many faces for processing. Will be processed only {} of {}".format(
                    self.QUEUE_SIZE, len(rois)
                )
            )
            rois = rois[: self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                if (
                    rois[i].position[0] == 0.0
                    or rois[i].position[1] == 0.0
                    or (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1])
                    or (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0])
                ):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(
                        crop_image, face_identities[i].descriptor, name
                    )
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]


def draw_detections(frame, frame_processor, detections, output_transform):
    def draw_detections_function(frame):
        size = frame.shape[:2]
        frame = output_transform.resize(frame)
        for roi, landmarks, identity in zip(*detections):
            text = frame_processor.face_identifier.get_identity_label(identity.id)
            if identity.id != FaceIdentifier.UNKNOWN_ID:
                text += " %.2f%%" % (100.0 * (1 - identity.distance))
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
            for point in landmarks:
                x = xmin + output_transform.scale(roi.size[0] * point[0])
                y = ymin + output_transform.scale(roi.size[1] * point[1])
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(
                frame,
                (xmin, ymin),
                (xmin + textsize[0], ymin - textsize[1]),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
            )

    draw_detections_thread = threading.Thread(
        target=draw_detections_function, args=(frame,)
    )
    draw_detections_thread.start()
    draw_detections_thread.join()


def save_face(detections, frame):
    def save_face_function():
        for idx, (roi, _, identity) in enumerate(zip(*detections)):
            if identity.id == FaceIdentifier.UNKNOWN_ID:
                # Get the coordinates of the bounding box
                xmin = max(int(roi.position[0]), 0)
                ymin = max(int(roi.position[1]), 0)
                xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
                ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])

                # Crop and save the face region
                face_region = frame[ymin:ymax, xmin:xmax]
                face_dir = current_path + "/gall/"
                file_pattern = re.compile(r"face_(\d+)\.jpg")
                existing_numbers = []
                for filename in os.listdir(face_dir):
                    match = file_pattern.match(filename)
                    if match:
                        existing_numbers.append(int(match.group(1)))

                # Find the highest number or initialize to 0
                if existing_numbers:
                    next_number = max(existing_numbers) + 1
                else:
                    next_number = 1

                # Generate the next file name
                new_filename = f"{face_dir}face_{next_number}.jpg"
                cv2.imwrite(new_filename, face_region)
                print(f"Saved face region as {new_filename}")

    save_face_thread = threading.Thread(target=save_face_function)
    save_face_thread.start()
    save_face_thread.join()


def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[
        (fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
        (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
        :,
    ]


if __name__ == "__main__":
    init()
    realsense()
