import cv2
import numpy as np

def apply_perspective_transform(image, src_points, dst_points, output_size=(720, 480)):
    """
    對給定的圖像應用透視變換。

    :param image_path: 圖像檔案的路徑
    :param src_points: 源點座標 (四個點)
    :param dst_points: 目標點座標 (將源點映射到的矩形區域)
    :param output_size: 輸出的圖像尺寸 (默認為 720x480)
    
    :return: 透視變換後的圖像
    """
    # 讀取圖像
    

    print(f"Original image size: {image.shape[0]}x{image.shape[1]}")

    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 進行透視變換
    warped_image = cv2.warpPerspective(image, M, output_size)

    return warped_image

# TODO: 將點擊座標值輸入到此
src_points = np.float32([
    (85, 807), # 左上
    (448, 655), # 右上 
    (681, 770), # 右下
    (234, 1050) # 左下
])

dst_points = np.float32([
    (0, 0),
    (720, 0),
    (720, 480),
    (0, 480)
])

image_path = '../image/coin01.png'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Unable to load image at {image_path}")
warped_image = apply_perspective_transform(image, src_points, dst_points)

# 顯示結果 (選擇性)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
