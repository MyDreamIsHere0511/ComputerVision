import sys
import math
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Cau hinh duong dan - dat o dau de de chinh sua
input_filename = 'ball1.jpg'  # Ten file anh dau vao
output_filename = 'result_lines.jpg'  # Ten file anh ket qua
figure_filename = 'result_combined.png'  # Ten file anh ket hop

# Ham tim anh
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

def detect_lines(image_path):
    src = cv.imread(image_path)
    if src is None:
        raise Exception(f"Khong the doc anh tu {image_path}")
        
    result_img = src.copy()
    
    if len(src.shape) == 3:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    blurred = cv.GaussianBlur(enhanced, (5, 5), 0)
    
    edges = cv.Canny(blurred, 30, 150, apertureSize=3)
    
    all_lines = []
    
    param_sets = [
        (1, np.pi/180, 25, 30, 5),
        (1, np.pi/180, 40, 35, 5),
        (1, np.pi/180, 60, 50, 3),
        (0.5, np.pi/360, 30, 25, 3)
    ]
    
    for rho, theta, threshold, min_length, max_gap in param_sets:
        lines = cv.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            lines=None,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length >= min_length:
                    all_lines.append(line)
    
    unique_lines = []
    tolerance = 10
    
    if all_lines:
        used = np.zeros(len(all_lines), dtype=bool)
        
        for i, line1 in enumerate(all_lines):
            if used[i]:
                continue
                
            x1, y1, x2, y2 = line1[0]
            length1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            used[i] = True
            best_line = line1
            best_length = length1
            
            for j, line2 in enumerate(all_lines[i+1:], i+1):
                if used[j]:
                    continue
                    
                x3, y3, x4, y4 = line2[0]
                
                d1 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
                d2 = np.sqrt((x2-x4)**2 + (y2-y4)**2)
                d3 = np.sqrt((x1-x4)**2 + (y1-y4)**2)
                d4 = np.sqrt((x2-x3)**2 + (y2-y3)**2)
                
                if min(d1+d2, d3+d4) < tolerance:
                    used[j] = True
                    
                    length2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                    if length2 > best_length:
                        best_length = length2
                        best_line = line2
            
            unique_lines.append(best_line)
    
    line_thickness = 2
    
    for line in unique_lines:
        l = line[0]
        cv.line(result_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), line_thickness, cv.LINE_AA)
    
    return src, result_img

def show_results_in_figure(original, result, title="Phat hien duong thang"):
    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.suptitle(title, fontsize=16)
    
    # Anh goc
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    plt.title('Anh goc')
    plt.axis('off')
    
    # Anh ket qua
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Phat hien duong thang')
    plt.axis('off')
    
    plt.tight_layout()
    
    return fig

def main(argv):
    # Su dung ten file tu dong lenh hoac tu cau hinh
    filename = argv[0] if len(argv) > 0 else input_filename
    
    img_path = find_image(filename)
    if not img_path:
        print(f"Khong tim thay file '{filename}'")
        return -1
    
    try:
        # Phat hien duong thang
        original, result = detect_lines(img_path)
        
        # Luu anh ket qua
        output_dir = os.path.dirname(img_path)
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(output_dir, output_filename)
        cv.imwrite(output_path, result)
        
        # Hien thi va luu hinh anh ket hop
        fig = show_results_in_figure(original, result, f"Phat hien duong thang - {filename}")
        figure_path = os.path.join(output_dir, figure_filename)
        fig.savefig(figure_path, dpi=100, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
    except Exception as e:
        print(f"Loi: {str(e)}")
        return -1
    
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
