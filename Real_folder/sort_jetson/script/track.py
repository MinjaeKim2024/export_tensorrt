import argparse
import torch
import cv2
import numpy as np
from utils import semi_yolo, on_predict_start, TrackerVisualizer
import logging
from functools import partial

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default=0,
                        help='video path, 0 for webcam')
    parser.add_argument('--det-thresh', type=float, default='0',
                        help='detection_threshold')
    parser.add_argument('--max-age', type=int, default='30',
                        help='max age for tracks')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.3,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--custom', action='store_true',
                        help='custom weights, i.e. True or False')
    parser.add_argument('--num-classes', type=int, default = 751,
                        help='number of classes in custom weights of reid model, i.e. 751(market1501 dataset)')
    parser.add_argument('--weights', type=str, default = "osnet_x1_25_msmt17.pt",
                        help='weights file, i.e. osnet_x1_25_msmt17.pt')
    parser.add_argument('--device', default='cuda',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--reid', action='store_true',
                        help='use reid model for tracking')
    parser.add_argument('--custom-detector', action='store_true',
                        help='use custom detector')
    # <---------------------------------------------------------------------------------------------------------------------------------------> #
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    
    # <---------------------------------------------------------------------------------------------------------------------------------------> #       

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    opt = parse_opt()
    device = opt.device
    
    if not opt.custom_detector:
        detector = semi_yolo()
        results = detector.track(
            source=opt.source,
            conf=opt.conf,
            iou=opt.iou,
            show=opt.show,
            stream=True,
            device=device,
            show_conf=opt.show_conf,
            save_txt=opt.save_txt,
            show_labels=opt.show_labels,
            verbose=opt.verbose,
            exist_ok=opt.exist_ok,
            name=opt.name,
            classes=opt.classes,
            imgsz=opt.imgsz,
            vid_stride=opt.vid_stride,
            line_width=opt.line_width
        )
        
        detector.add_callback('on_predict_start', partial(on_predict_start, reid = opt.reid,
                                                      weights = opt.weights, device = device,
                                                      custom = opt.custom, max_age = opt.max_age,
                                                      iou = opt.iou, det_thresh = opt.det_thresh,
                                                      num_classes = opt.num_classes, custom_detector = opt.custom_detector))
        detector.predictor.custom_args = opt
        for idx, r in enumerate(results):
            pass
    else:
        # we don't have custom model now, so below is just for debugging
        # --------------------------Delete below and Put your detection model here --------------------------------- #
        # dummy model
        def dummy_detector(x):
            return np.array([[1, 50, 30, 480, 0.82, 0],[425, 281, 576, 472, 0.56, 65]]) # return type : (x, y, x, y, conf, cls)
        detector = dummy_detector
        
        # -------------------------------------------------------------------------------------------------- #
        tracker = on_predict_start(predictor= None, reid = opt.reid,
                                    weights = opt.weights, device = device,
                                    custom = opt.custom, max_age = opt.max_age,
                                    iou = opt.iou, det_thresh = opt.det_thresh,
                                    num_classes = opt.num_classes, custom_detector = opt.custom_detector)
        
        vid = cv2.VideoCapture(opt.source)    
        ret = True
        visualizer = TrackerVisualizer()
        
        while ret:
            print("reading video")
            ret, im = vid.read()
            
            im_resized = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
            import torch.nn.functional as F
            im_tensor = (torch.from_numpy(im_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)) / 255.0
            
            # dets = (x, y, x, y, conf, cls)
            results = detector(im)
            dets = []
            for result in results:
                xyxy = result[:4]
                conf = result[4]
                cls_ = result[5] 
                det = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, int(cls_)]
                dets.append(det)
            dets = np.array(dets)
            if dets.size == 0:
                dets = np.zeros((1,6), dtype=int)

            tracks = tracker.update(dets, im_resized) # --> (x, y, x, y, id, conf, cls, ind), (144, 212, 578, 480, 1, 0.82, 0, 0)
            
            # Assume 'im', 'tracks', and 'opt' are defined
            should_quit = visualizer.display_tracking_results(im, tracks, opt)
            # Display tracking results
            if opt.show :
                should_quit = visualizer.display_tracking_results(im, tracks, opt)
                if should_quit:
                    break
            
        vid.release()
        if opt.show:
            cv2.destroyAllWindows()
        