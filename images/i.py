import cv2
import os
import matplotlib.pyplot as plt

def show_image_resolution(image_path):
    # Kiểm tra xem file tồn tại không
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Lấy kích thước ảnh (độ phân giải)
    height, width, channels = img.shape
    
    # Hiển thị thông tin
    print(f"Đường dẫn ảnh: {image_path}")
    print(f"Độ phân giải: {width} x {height} pixels")
    print(f"Số kênh màu: {channels}")
    
    # Hiển thị ảnh
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Ảnh: {os.path.basename(image_path)} - Độ phân giải: {width} x {height}")
    plt.axis('off')
    plt.show()

# Sử dụng hàm với các ảnh trong tài liệu của bạn
show_image_resolution("hi.jpg")
show_image_resolution("1.jpg")