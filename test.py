"""
===================================================================================
TEST SCRIPT - KIỂM TRA ĐẦU VÀO VÀ ĐẦU RA CỦA MÔ HÌNH AMODALSWINUNET
===================================================================================
Script này thực hiện các công việc sau:
1. Tải một mẫu ngẫu nhiên từ AmodalDataset.
2. Tách và hiển thị các thành phần đầu vào của mô hình:
   - Ảnh RGB
   - Visible Mask (phần nhìn thấy)
   - Edge Mask (viền của phần nhìn thấy)
3. Chạy mô hình AmodalSwinUNet để lấy kết quả dự đoán.
4. Hiển thị song song Ground Truth (sự thật) và kết quả dự đoán để so sánh.

Cách chạy:
1. Đảm bảo bạn đã có thư mục `data` và `checkpoints`.
2. Chạy script từ thư mục gốc của dự án: `python test.py`
===================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random

# Giả định script này nằm ở thư mục gốc của dự án
from scripts.dataset import AmodalDataset
from scripts.model import AmodalSwinUNet


def run_test():
    """
    Hàm chính để tải mô hình, dữ liệu và thực hiện kiểm tra.
    """
    # ──────────────────────────────────────────────────────────────────
    # CẤU HÌNH
    # ──────────────────────────────────────────────────────────────────
    IMG_DIR = 'data/val2014'
    ANN_FILE = 'data/annotations/COCO_amodal_val2014.json'
    CHECKPOINT_PATH = 'checkpoints/swin_amodal_epoch_30.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Sử dụng thiết bị: {DEVICE}")

    # ──────────────────────────────────────────────────────────────────
    # 1. TẢI MÔ HÌNH
    # ──────────────────────────────────────────────────────────────────
    print("📥 Đang nạp mô hình Swin-UNet...")
    model = AmodalSwinUNet(num_classes=91).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()  # Chuyển sang chế độ đánh giá
    print("✅ Nạp mô hình thành công!")

    # ──────────────────────────────────────────────────────────────────
    # 2. TẢI DATASET
    # ──────────────────────────────────────────────────────────────────
    # Mô hình yêu cầu input 224x224, ta cần resize ảnh và mask
    transform = A.Compose([
        A.Resize(224, 224),
    ])

    dataset = AmodalDataset(img_dir=IMG_DIR, ann_file=ANN_FILE, transform=transform)
    print(f"✅ Nạp dataset thành công với {len(dataset)} mẫu vật thể.")

    # ──────────────────────────────────────────────────────────────────
    # 3. LẤY MẪU, DỰ ĐOÁN VÀ HIỂN THỊ
    # ──────────────────────────────────────────────────────────────────
    # Lấy một index ngẫu nhiên
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"\n🎨 Đang chuẩn bị hiển thị mẫu số: {sample_idx}")

    # Lấy dữ liệu từ dataset
    input_tensor, amodal_gt_tensor, _, cat_id_tensor = dataset[sample_idx]

    # Chuẩn bị dữ liệu cho mô hình (thêm chiều batch)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    cat_id_batch = cat_id_tensor.unsqueeze(0).to(DEVICE)

    # Chạy dự đoán
    with torch.no_grad():
        logits = model(input_batch, cat_id_batch)
        # Áp dụng Sigmoid để có xác suất và phân ngưỡng 0.5 để có mask nhị phân
        pred_mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # Tách các kênh từ input_tensor để hiển thị
    # input_tensor có shape [5, H, W]
    rgb_img = input_tensor[:3, :, :].permute(1, 2, 0).numpy() # Chuyển về [H, W, C] cho matplotlib
    visible_mask = input_tensor[3, :, :].numpy()
    edge_mask = input_tensor[4, :, :].numpy()
    ground_truth_mask = amodal_gt_tensor.numpy()

    # Vẽ kết quả
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Kiểm Tra Mẫu #{sample_idx} - Class ID: {cat_id_tensor.item()}', fontsize=16)

    axes[0].imshow(rgb_img)
    axes[0].set_title("1. RGB")
    axes[1].imshow(visible_mask, cmap='gray')
    axes[1].set_title("2. Visible Mask")
    axes[2].imshow(edge_mask, cmap='gray')
    axes[2].set_title("3. Edge Mask")
    axes[3].imshow(ground_truth_mask, cmap='gray')
    axes[3].set_title("4. Ground Truth (Amodal)")
    axes[4].imshow(pred_mask, cmap='gray')
    axes[4].set_title("5. Prediction")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_test()