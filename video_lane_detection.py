import torch
import cv2
import argparse
import os
import torchvision.transforms as transforms
from utils.common import merge_config, get_model
from demo import pred2coords
from utils.config import Config
import numpy as np


def main():
    # 参数解析（添加config参数）
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
        车道线检测视频处理工具
        示例命令：
        python video_lane_detection.py \\
            --config configs/tusimple_res18.py \\
            --input input_video.mp4 \\
            --output output_processed.mp4 \\
            --model weights/tusimple_res18.pth

        ''')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径（如：configs/tusimple_res18.py）')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径（支持MP4/AVI格式）')
    parser.add_argument('--output', type=str, required=True, help='输出视频路径（MP4格式）')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型权重路径（如：weights/tusimple_res18.pth）')
    args = parser.parse_args()

    # 加载配置文件（自动继承tusimple_res18.py所有参数）
    #_, cfg = merge_config(args.config)
    cfg = Config.fromfile(args.config)
    
    cfg.test_model = args.model  # 覆盖模型路径
    cfg.batch_size = 1

    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    # 初始化模型（自动使用配置文件中的参数）
    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    print(cfg.train_height, cfg.train_width, cfg.crop_ratio)

    # 图像预处理（使用配置文件参数）
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_w, img_h = 1280, 720

    # 打开视频流
    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # 获取视频参数
    target_fps = 2.0  # 目标处理帧率
    process_interval = max(1, int(round(fps / target_fps)))  # 计算跳帧间隔    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, target_fps, (img_w, img_h))
    out0 = cv2.VideoWriter(args.output+'_0.mp4', fourcc, target_fps, (img_w, img_h))

    # 车道线颜色配置（与demo.py相同）
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160,710, cfg.num_row)/720
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    # 锚点配置应从数据集常量获取（推荐方式）
    row_anchor = cfg.row_anchor
    col_anchor = cfg.col_anchor


    # 在视频处理循环中：
    frame_count = 0
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count < 30 * 60 * 3:
                frame_count += 1
                continue
            # 仅处理满足间隔条件的帧
            if frame_count % process_interval == 0:
                #print("frame_count:", frame_count)
                # 预处理
                frame = cv2.resize(frame, (img_w, img_h))
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                out0.write(frame)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = img_transforms(img_rgb)
                img = img[:,-cfg.train_height:,:]
                img = img.unsqueeze(0).cuda()
                #img = img_transforms(img_rgb)
                #print(img.shape)

                # 推理
                pred = net(img)
                
                # 转换坐标
                coords = pred2coords(pred, row_anchor, col_anchor,
                                   original_image_width=img_w,
                                   original_image_height=img_h)
                
                # 绘制车道线
                for lane_idx, lane in enumerate(coords):
                    color = colors[lane_idx % len(colors)]
                    for coord in lane:
                        cv2.circle(frame, coord, 5, color, -1)

                out.write(frame)
                #cv2.imwrite("./tmp/"+args.output+'_'+str(frame_count)+'.jpg', frame)
            frame_count += 1


            if frame_count > 30 * 60 * 10:
                break
    cap.release()
    out.release()
    out0.release()
    print(f"处理完成，结果保存至: {args.output}")

if __name__ == "__main__":
    #python video_lane_detection.py --config configs/tusimple_res18.py  --model weights/pretrained/tusimple_res18.pth   --output output_processed.mp4 --input ./datasets/route28.mp4
           
    main() 