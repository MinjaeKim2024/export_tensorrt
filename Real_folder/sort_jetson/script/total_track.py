
import torch
import cv2
import numpy as np
from utils import TrackerVisualizer
import logging
from deepocsort_model import DeepOCSort
import os
from args import make_parser
logging.disable(logging.INFO) 

# argument section 
def get_main_args():
    parser = make_parser()
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="run post-processing linear interpolation.",
    )
    parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument(
        "--alpha_fixed_emb",
        type=float,
        default=0.95,
        help="Alpha fixed for EMA embedding",
    )
    parser.add_argument("--emb_off", action="store_true")
    parser.add_argument("--cmc_off", action="store_true")
    parser.add_argument("--aw_off", action="store_true")
    parser.add_argument("--aw_param", type=float, default=0.5)
    parser.add_argument("--new_kf_off", action="store_true")
    parser.add_argument("--grid_off", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument('--r_weights', type=str, default = "weights/osnet_ain_x1_0_dukemtmcreid.engine",
                        help='weights file, i.e. weights/osnet_ain_x1_0_dukemtmcreid.engine')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--source', type=str, default=0,
                        help='video path, 0 for webcam')
    
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    

    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "dance":
        args.result_folder = os.path.join(args.result_folder, "DANCE-val")
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
        
    
    device = 'cuda'
    
    # args
    args = get_main_args()
    oc_sort_args = dict(
        args=args,
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
        w_association_emb=args.w_assoc_emb,
        alpha_fixed_emb=args.alpha_fixed_emb,
        embedding_off=args.emb_off,
        cmc_off=args.cmc_off,
        aw_off=args.aw_off,
        aw_param=args.aw_param,
        new_kf_off=args.new_kf_off,
        grid_off=args.grid_off,
    )
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("weights/yolov8n.engine")  # load a pretrained model (recommended for training)
    

    from pathlib import Path
    tracker = DeepOCSort(model_weights=Path(args.r_weights), device=device, fp16 = False,**oc_sort_args)

    vid = cv2.VideoCapture(args.source)    

    ret = True
    visualizer = visualizer = TrackerVisualizer(save_path="result.mp4" if args.save else None, 
                                                frame_size=(640, 480),
                                                fps=20.0)
    
    
    while ret:
        ret, im = vid.read()
        if not ret:
            visualizer.release()
        # im: (h, w, c), c: (b, g, r)
        results = model(im,conf=0.7)[0]
        det = results.boxes.data.cpu().numpy()
        target = []
        
        # skip other class
        for ids, ii in enumerate(det):
            if ii[5] == 0:
                target.append(ids)
        det = det[target]
        results = results[target]
        
        final_result = []
        if len(det) == 0:
            if args.show:
                if visualizer.display_tracking_results(im, final_result, args):
                    break
            # continue
        tracks = tracker.update(det, im)
        
        if len(tracks) == 0:
            continue
        
        idx = tracks[:, -1].astype(int)
        results = results[idx]
        results.update(boxes=torch.as_tensor(tracks[:, :-1]))
        
        for result in results:
            xyxy = result.boxes.xyxy[0]
            id_ = result.boxes.id
            conf = result.boxes.conf
            cls_ = result.boxes.cls
            xcl = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), id_.item(), conf.item(), int(cls_)]
            final_result.append(xcl)
        final_result = np.array(final_result)
        
        # Display tracking results
        if args.show :
            if visualizer.display_tracking_results(im, final_result, args):
                break
        
        
    if args.show:
        cv2.destroyAllWindows()
    