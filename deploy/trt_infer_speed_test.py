import cv2
import time
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config



class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        cfg = Config.fromfile(config_path)
        self.ori_img_w, self.ori_img_h = ori_size
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def forward_core(self, img):
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        preds = {}
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            preds[out['name']] = output
        return preds



def test_fps(engine_path, config_path, ori_size, test_seconds=30):
    # 初始化模型
    model = UFLDv2(engine_path, config_path, ori_size)
    
    # 生成并预处理虚拟图像
    dummy_img = np.zeros((ori_size[1], ori_size[0], 3), dtype=np.uint8)
    # 预处理步骤（只执行一次）
    processed_img = dummy_img[model.cut_height:, :, :]
    processed_img = cv2.resize(processed_img, (model.input_width, model.input_height), cv2.INTER_CUBIC)
    processed_img = processed_img.astype(np.float32) / 255.0
    processed_img = np.transpose(np.float32(processed_img[:, :, :, np.newaxis]), (3, 2, 0, 1))
    processed_img = np.ascontiguousarray(processed_img)

    # 预热运行
    print("Warming up...")
    for _ in range(100):
        model.forward_core(processed_img)
    
    # 正式测试
    print("Start benchmarking core inference...")
    frame_count = 0
    start_time = time.time()
    
    while (time.time() - start_time) < test_seconds:
        model.forward_core(processed_img)
        frame_count += 1
    
    # 计算FPS
    total_time = time.time() - start_time
    fps = frame_count / total_time
    print(f"Core inference FPS: {fps:.2f}")
    return fps

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py', 
                      help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weights/culane_res34.engine',
                      help='path to engine file', type=str)
    parser.add_argument('--ori_size', default=(1280, 720), 
                      help='size of original frame', type=tuple)
    parser.add_argument('--test_seconds', type=int, default=30,
                      help='duration of the test in seconds')
    return parser.parse_args()

if __name__ == "__main__":
    # 命令行example: 
    # python ./deploy/trt_infer_speed_test.py --config_path configs/tusimple_res18.py --engine_path weights/pretrained/tusimple_res18.engine --test_seconds 10
    # For a fp32 model, on a PC with RTX4090, the FPS is about 800.
    # For a fp32 model, on a Jetson Orin NX 16GB, the FPS is about 50.
    args = get_args()
    test_fps(
        args.engine_path,
        args.config_path,
        args.ori_size,
        args.test_seconds
    ) 