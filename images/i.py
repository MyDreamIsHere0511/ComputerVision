import cv2
import os
import matplotlib.pyplot as plt
import time
from tabulate import tabulate

def get_image_resolution(image_path):
    """
    Trả về thông tin độ phân giải của ảnh
    
    Args:
        image_path: Đường dẫn đến file ảnh
    
    Returns:
        Tuple (width, height, channels) hoặc None nếu không đọc được ảnh
    """
    # Kiểm tra xem file tồn tại không
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return None
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Lấy kích thước ảnh (độ phân giải)
    height, width, channels = img.shape
    
    return (width, height, channels, img)

def show_all_images_resolution(image_files):
    """
    Hiển thị thông tin độ phân giải của nhiều ảnh và hiển thị chúng
    
    Args:
        image_files: List các đường dẫn ảnh cần kiểm tra
    """
    # Tạo danh sách lưu thông tin cho bảng
    table_data = []
    valid_images = []
    
    # Thu thập thông tin về từng ảnh
    for img_path in image_files:
        result = get_image_resolution(img_path)
        if result:
            width, height, channels, img = result
            file_size_kb = os.path.getsize(img_path) / 1024  # Kích thước file (KB)
            table_data.append([
                os.path.basename(img_path),
                f"{width} x {height}",
                channels,
                f"{file_size_kb:.2f} KB"
            ])
            valid_images.append((img_path, img, width, height))
    
    # In thông tin dưới dạng bảng
    print("\n" + "="*60)
    print("THÔNG TIN ĐỘ PHÂN GIẢI CỦA CÁC ẢNH")
    print("="*60)
    
    headers = ["Tên file", "Độ phân giải", "Kênh màu", "Kích thước file"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Hiển thị các ảnh với thông tin
    if valid_images:
        # Tính số hàng và cột để hiển thị ảnh
        n_images = len(valid_images)
        cols = min(3, n_images)  # Tối đa 3 ảnh mỗi hàng
        rows = (n_images + cols - 1) // cols
        
        # Tạo subplot
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]  # Đảm bảo axes luôn là mảng 2D
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        # Hiển thị từng ảnh
        for i, (img_path, img, width, height) in enumerate(valid_images):
            row, col = i // cols, i % cols
            ax = axes[row][col]
            
            # Chuyển BGR sang RGB để hiển thị đúng với matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            
            # Tiêu đề với thông tin ảnh
            filename = os.path.basename(img_path)
            ax.set_title(f"{filename}\n{width} x {height}")
            ax.axis('off')
        
        # Ẩn các subplot không sử dụng
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Danh sách các file ảnh cần kiểm tra
    image_files = [
        "ball2.png",
        "plant.webp",  # Chú ý: trong đầu bài là plant.webp (không phải planet.webp)
        "circle1.jpg",
        "hinhkhoi.jpg", 
        "ball1.jpg",
        "sudoku.jpg"
    ]
    
    # Kiểm tra xem các file có đúng trong thư mục hiện tại không
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(len(image_files)):
        # Kiểm tra trong thư mục hiện tại
        if os.path.exists(image_files[i]):
            continue
        
        # Kiểm tra trong thư mục Picture
        pic_path = os.path.join(current_dir, "Picture", image_files[i])
        if os.path.exists(pic_path):
            image_files[i] = pic_path
            continue
            
        # Kiểm tra trong thư mục chứa images (giả định từ repo GitHub)
        repo_path = os.path.join(current_dir, "..", "images", "Picture", image_files[i])
        if os.path.exists(repo_path):
            image_files[i] = repo_path
    
    # Hiển thị thông tin độ phân giải
    show_all_images_resolution(image_files)