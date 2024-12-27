import cv2
import numpy as np

# 回呼函數：用來更新圖片
def update_threshold(x):
    # 取得滑塊的值
    h_min = cv2.getTrackbarPos('H Min', 'Filtered Image')
    h_max = cv2.getTrackbarPos('H Max', 'Filtered Image')
    s_min = cv2.getTrackbarPos('S Min', 'Filtered Image')
    s_max = cv2.getTrackbarPos('S Max', 'Filtered Image')
    v_min = cv2.getTrackbarPos('V Min', 'Filtered Image')
    v_max = cv2.getTrackbarPos('V Max', 'Filtered Image')

    # 根據滑塊的值創建顏色範圍
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # 過濾顏色
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 顯示結果
    cv2.imshow('Filtered Image', filtered_image)

# 載入圖片
image = cv2.imread('../image/coin1_trans_v1.png')  # 更換為你的圖片路徑
# contrast = 200
# brightness = 0
# precess_img = image * (contrast/127 + 1) - contrast + brightness
# precess_img = 255 - precess_img
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 創建一個窗口顯示過濾後的圖片
cv2.namedWindow('Filtered Image')

# 設定滑塊來調整顏色範圍
cv2.createTrackbar('H Min', 'Filtered Image', 0, 179, update_threshold)
cv2.createTrackbar('H Max', 'Filtered Image', 179, 179, update_threshold)
cv2.createTrackbar('S Min', 'Filtered Image', 0, 255, update_threshold)
cv2.createTrackbar('S Max', 'Filtered Image', 255, 255, update_threshold)
cv2.createTrackbar('V Min', 'Filtered Image', 0, 255, update_threshold)
cv2.createTrackbar('V Max', 'Filtered Image', 255, 255, update_threshold)

# 初始化顯示
update_threshold(0)

# 持續顯示直到按 'q' 鍵
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
