import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('../image/test.png')  # 確保 'image.png' 是您的圖像檔案
print(image.shape[0], image.shape[1])
# image = cv2.resize(image , (720, 480), interpolation=cv2.INTER_AREA)

# 定義源點 (您提供的四個點)
src_points = np.float32([
    (85, 807),
    (448, 655),
    (681, 770),
    (234, 1050)
])

# (85, 807)
# (448, 655)
# (681, 770)
# (234, 997)
# (351, 992)

# 定義目標點 (將源點映射到的矩形區域)
dst_points = np.float32([
    (0, 0),
    (720, 0),
    (720, 480),
    (0, 480)
])

# 計算透視變換矩陣
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 進行透視變換
warped_image = cv2.warpPerspective(image, M, (720, 480))
warped_image_resized = cv2.resize(warped_image, (720, 480), interpolation=cv2.INTER_CUBIC)
# 顯示結果
cv2.imshow('Original Image00', image)
cv2.imshow('Warped Image00', warped_image)
cv2.imwrite('transfer_coin_test.png', warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()