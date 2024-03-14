# DeepOCSORT + OSNet


## Installation
```
git clone https://github.com/RGA-Robot/AIP_Object-tracking.git
cd custom_ocsort
pip install -r requirements.txt
```

- Make sure you have torch compatible with your cuda version


## Custom weight file for Re-ID model

```
# Ensure that the name of one of your weight files contains one of the following strings:
'''
    "resnet50"
    "resnet101"
    "mlfn"
    "hacnn"
    "mobilenetv2_x1_0"
    "mobilenetv2_x1_4"
    "osnet_x1_0"
    "osnet_x0_75"
    "osnet_x0_5"
    "osnet_x0_25"
    "osnet_ibn_x1_0"
    "osnet_ain_x1_0"
    "lmbn_n"
    "clip"

'''
```
- This decides your ReID model
- Put your weight file in
    +  AIP_Object_tracking/custom_ocsort/

## How to use custom detection model
- script/track.py (L103 - L109)
```
# -------- Delete below and Put your detection model here ---------- #
        # dummy model
        def dummy_detector(x):
            return np.array([[1, 50, 30, 480, 0.82, 0],[425, 281, 576, 472, 0.56, 65]]) # return type : (x, y, x, y, conf, cls)
        detector = dummy_detector
# ------------------------------------------------------------------ #
        
```
- Please put your detector in this part

## Inference Option

```
python3 script/total_track.py --custom --r_weights=weights/osnet_ain_x1_0_dukemtmcreid.engine --iou_thresh 0.1 --source output12.mp4 --track_thresh 0.75 --min_hits 10 --w_assoc_emb 0.99 --show

```

- option you can choose
    - --source: video or webcam # default is webcam (0)
    - --show: show in Online
    - --save: save video
    - --r_weights: path of your Re-ID weight file
    
    - --custom: if you use your custom ReID weight, please use this option and change number of classes
    - --num-classes: number of class in your custom dataset(Re-Id dataset)

    
    