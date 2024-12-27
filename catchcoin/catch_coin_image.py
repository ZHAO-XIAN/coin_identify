import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('../image/transfer_coin_v3.png')
output = image.copy()


# 轉換為灰階圖像
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)
# 高斯模糊來去除噪點
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

ret, blurred_threshold = cv2.threshold(gray_blurred, 100, 255, cv2.THRESH_BINARY)

kernel = np.ones((11, 11), np.uint8) #結構元素(kernel)
closing = cv2.morphologyEx(blurred_threshold, cv2.MORPH_CLOSE, kernel)

# 使用霍夫變換來檢測圓形
circles = cv2.HoughCircles(
    closing,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=30
)

# 繪製檢測到的圓形並標記代號
if circles is not None:
    circles = np.uint16(np.around(circles))
    for idx, i in enumerate(circles[0, :]):
        # 畫圓形輪廓
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 標記圓心
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
        # 標記代號 (C1, C2, C3...)
        label = f"C{idx + 1}"
        cv2.putText(output, label, (i[0] - 20, i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        print(label, "圓心:", (i[0], i[1]),"半徑:", i[2])

# 顯示結果
cv2.imshow('Detected Coins with Labels', output)
cv2.imshow("closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
