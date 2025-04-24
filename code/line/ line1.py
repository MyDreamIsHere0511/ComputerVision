import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Cau hinh duong dan
input_filename = 'sudoku.jpg'
output_filename = 'sudokutest.jpg'
figure_filename = 'lines_comparison.png'

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

# Thiet lap encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Doc anh
image_path = find_image(input_filename)
if not image_path:
    exit(1)

original = cv.imread(image_path)
if original is None:
    exit(1)

# Chuan bi xu ly
img = original.copy()
output_dir = os.path.dirname(image_path)
output_path = os.path.join(output_dir, output_filename)
figure_path = os.path.join(output_dir, figure_filename)

# Xu ly anh
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize = 3)

try:
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
except Exception:
    exit(1)

# Luu va hien thi ket qua
cv.imwrite(output_path, img)

# Hien thi voi matplotlib
plt.figure(figsize=(12, 6))

# Anh goc
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.title('Anh goc')
plt.axis('off')

# Anh ket qua
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Phat hien duong thang')
plt.axis('off')

plt.tight_layout()
plt.savefig(figure_path, dpi=100, bbox_inches='tight')
plt.show()