import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.maniqa import MANIQA


def parse_args():
    parser = argparse.ArgumentParser(description="NR-IQA Inference for a Single Image")
    parser.add_argument('--img_path', type=str, default='./test.jpg',
                        help='Path to the input distorted image.')
    parser.add_argument('--weight_path', type=str, default='./weights/best_model.pt',
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference (cuda or cpu).')
    return parser.parse_args()


def load_image(img_path, img_size=224):

    img = Image.open(img_path).convert('RGB')

    # 使用 CenterCrop 保持图像的原始失真频率分布
    transform = transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def main():
    args = parse_args()
    print(f"🚀 Initialize Inference on device: {args.device.upper()}")

    # 1. 实例化模型
    net = MANIQA(
        embed_dim=768,
        num_outputs=1,
        dim_mlp=768,
        patch_size=8,
        img_size=224,
        window_size=4,
        depths=[2, 2],
        num_heads=[4, 4],
        num_tab=2,
        scale=0.1
    )

    net = net.to(args.device)

    # 2. 加载预训练权重
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"⚠️ Model weight not found at {args.weight_path}")

    print(f"📦 Loading pre-trained weights from: {args.weight_path}")

    state_dict = torch.load(args.weight_path, map_location=args.device, weights_only=True)

    # 清洗 DataParallel 带来的 'module.' 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    # 提取 model_state_dict
    if 'model_state_dict' in clean_state_dict:
        net.load_state_dict(clean_state_dict['model_state_dict'], strict=True)
    else:
        net.load_state_dict(clean_state_dict, strict=True)

    net.eval()

    # 3. 读取图像并预测
    print(f"🖼️ Processing image: {args.img_path}")
    img_tensor = load_image(args.img_path, img_size=224).to(args.device)

    with torch.no_grad():
        score = net(img_tensor)

    # 获取纯数值
    predicted_quality = score.item()

    print("-" * 50)
    print(f"✅ Prediction Complete!")
    print(f"📈 Predicted Quality Score: {predicted_quality:.4f}")
    print("-" * 50)


if __name__ == '__main__':
    main()