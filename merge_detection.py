
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import torch
from pathlib import Path
import numpy as np

# yolo
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                        increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                letterbox, mixup, random_perspective)
from BinOpenpose import pyopenpose as op

def run(        
        weights = ROOT / 'weights/yolov5s.pt',  # model.pt path(s)
        source = ROOT / 'input/videos/railway-construction-site.avi',  # file/dir/URL/glob, 0 for webcam
        data = ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz = (640, 640),  # inference size (height, width)
        conf_thres = 0.25,  # confidence threshold
        iou_thres = 0.45,  # NMS IOU threshold
        max_det = 1000,  # maximum detections per image
        device = '',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = True,  # show results
        save_txt = True,  # save results to *.txt
        save_conf = False,  # save confidences in --save-txt labels
        save_crop = False,  # save cropped prediction boxes
        nosave = False,  # do not save images/videos
        classes = None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False,  # class-agnostic NMS
        augment = False,  # augmented inference
        visualize = False,  # visualize features
        update = False,  # update all models
        project = ROOT / 'output/detect',  # save results to project/name
        name = 'exp',  # save results to project/name
        exist_ok = False,  # existing project/name ok, do not increment
        line_thickness = 3,  # bounding box thickness (pixels)
        hide_labels = False,  # hide labels
        hide_conf = False,  # hide confidences
        half = True,  # use FP16 half-precision inference
        dnn = True,  # use OpenCV DNN for ONNX inference
        vid_stride = 1,  # video frame-rate stride):
        # transforms = False; 
        op_folder = "OpenposeModels"  # bin folder name of openpose models
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Starting OpenPose
    op_params = dict()
    op_params["model_folder"] = op_folder
    opWrapper = op.WrapperPython()
    opWrapper.configure(op_params)
    opWrapper.start()
    # Starting yolo
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # op init
    datum = op.Datum()
    cap = cv2.VideoCapture(source)
    # cap = cv2.VideoCapture("./input/videos/railway-construction-site.avi")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in this video: ' + str(framecount))
    videoWriter = cv2.VideoWriter("./output/op720_2.avi", cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    # yolo init
    bs = 1
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            frame_res = datum.poseKeypoints
            
            # create_mask()
            
            # yolo inferencing--------
            # preprocessing img
            # im = transforms(im0)
            im = letterbox(frame, imgsz, stride=stride, auto=pt)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            # inferencing
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
            # yolo process predictions--------
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            
            # visualization & output
            opframe = datum.cvOutputData
            cv2.imshow("Site Danger Detection based on OpenPose 1.7.0 & YOLOv5s", opframe)
            videoWriter.write(opframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'input/videos/railway-construction-site.avi', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'output/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', default=True, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=True, action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # parser.add_argument('--transforms', default=False, action='store_true', help='transform to img')
    parser.add_argument('--op-folder', type=str, default="OpenposeModels", help='bin folder name of openpose models')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # try:
    #     # Import Openpose (Windows/Ubuntu/OSX)
    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    #     try:
    #         # Windows Import
    #         # if platform == "win32":
    #             # Change these variables to point to the correct folder (Release/x64 etc.)
    #             # sys.path.append(dir_path + '/../../python/openpose/Release');
    #             # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
    #             # from BinOpenpose import pyopenpose as op
    #         # else:
    #             # Change these variables to point to the correct folder (Release/x64 etc.)
    #             # sys.path.append('../python');
    #             # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
    #             # sys.path.append('/usr/local/python')
    #             # from openpose import pyopenpose as op
    #     except ImportError as e:
    #         print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    #         raise e
    try:
        check_requirements(exclude=('tensorboard', 'thop'))
        run(**vars(opt))
        
    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)