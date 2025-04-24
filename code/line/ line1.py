import cv2 as cv
import numpy as np
import os
import sys

# Cấu hình đường dẫn
input_filename = 'sudoku.jpg'
output_filename = 'sudokutest.jpg'

# Hàm tìm ảnh
def find_image(filename):
    search_locations = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.abspath(os.curdir)
    ]
    
    for location in search_locations:
        path = os.path.join(location, filename)
        if os.path.isfile(path):
            return path
    return None

# Thiết lập encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Đọc ảnh
image_path = find_image(input_filename)
if not image_path:
    print(f"Error: File '{input_filename}' not found")
    exit(1)

img = cv.imread(image_path)
if img is None:
    print(f"Error: Cannot read image file")
    exit(1)

output_path = os.path.join(os.path.dirname(image_path), output_filename)

# Xử lý ảnh
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize = 3)

try:
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
except Exception as e:
    print(f"Error processing image: {e}")
    exit(1)

# Lưu và hiển thị kết quả
cv.imwrite(output_path, img)
cv.imshow('Detected Lines', img)
cv.waitKey(0)
cv.destroyAllWindows()