import cv2
import time
import argparse
from trt_infer import UFLDv2  # 复用已有的推理类
import numpy as np


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
    args = get_args()
    test_fps(
        args.engine_path,
        args.config_path,
        args.ori_size,
        args.test_seconds
    ) 