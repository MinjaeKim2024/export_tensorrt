
from collections import OrderedDict, namedtuple
from utils import logger as LOGGER
import cv2
import numpy as np
import tensorrt as trt
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages')
import pycuda.driver as cuda
import pycuda.autoinit 

class ReIDDetectMultiBackend:
    def __init__(self, weights="osnet_x0_25_msmt17.engine", fp16=False, custom=False, num_classes=751):
        super().__init__()

        self.weights = weights
        self.fp16 = fp16
        self.custom = custom
        self.num_classes = num_classes

        # 로거 설정
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        self.load_engine(TRT_LOGGER)

    def load_engine(self, logger):
        # TensorRT init, engine load
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        with open(self.weights, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        
        self.bindings = OrderedDict()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(index)))
            shape = self.engine.get_binding_shape(index)
            self.bindings[name] = {'dtype': dtype, 'shape': shape}

            # FP16 설정 확인
            if self.engine.get_binding_dtype(index) == trt.DataType.HALF:
                self.fp16 = True

    def preprocess(self, xyxys, img, fp16=False):
        crops = []
        h, w = img.shape[:2]
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            crops.append(crop)

        crops = np.stack(crops, axis=0)
        crops = np.transpose(crops, (0, 3, 1, 2))  # NCHW 형식으로 변환

        if fp16:
            crops = crops.astype(np.float16)
        else:
            crops = crops.astype(np.float32)

        return crops

    def forward(self, im_batch):
        # numpy copy to GPU
        input_binding_idx = self.engine.get_binding_index("images")
        d_input = cuda.mem_alloc(im_batch.nbytes)
        ch = im_batch.shape[0]
        im_batch_contiguous = np.ascontiguousarray(im_batch)
        cuda.memcpy_htod(d_input, im_batch_contiguous)
        if not self.context.set_binding_shape(input_binding_idx, im_batch.shape):
        #    raise ValueError("Failed to set binding shape for the input tensor.")
            pass
        
        output_binding_idx = self.engine.get_binding_index("output")
        
        # transform into python tuple
        output_shape_tuple = tuple([ch, 512])
        dtype = self.bindings['output']['dtype']
        # calculate bite size
        output_size = ch * 512 * dtype.itemsize
        d_output = cuda.mem_alloc(output_size)
        
        # binding addr
        bindings = [None] * self.engine.num_bindings
        bindings[input_binding_idx] = int(d_input)
        bindings[output_binding_idx] = int(d_output)

        # execute
        self.context.execute_v2(bindings=bindings)

        # gpu result to device
        output = np.empty(output_shape_tuple, dtype=dtype)
        cuda.memcpy_dtoh(output, d_output)

        # optional(this part I don't know it's necessary or not)
        d_input.free()
        d_output.free()
        return output

    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.preprocess(xyxys, img)
            features = self.forward(crops)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)
        
        return features
