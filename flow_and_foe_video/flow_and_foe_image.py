import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(x, y, foe_x, foe_y):
    return np.sqrt((x - foe_x)**2 + (y - foe_y)**2)

def calculate_optical_flow(image1_path, image2_path, magnitude_std_thresh=0):
    frame1 = cv2.imread(image1_path)
    frame2 = cv2.imread(image2_path)
    
    roi_x, roi_y, roi_w, roi_h = 0, 640, 600, 500 
    
    frame1 = frame1[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    frame2 = frame2[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    if frame1 is None or frame2 is None:
        print("Error: Unable to read one or both images.")
        return None, None, None

    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_frame1,
        gray_frame2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=50,  # 控制平滑程度
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    # 繪製光流圖
    flow_image = frame1.copy()  # 複製第一張影像用於繪製光流
    h, w, _ = flow_image.shape
    for i in range(0, h, 10):
        for j in range(0, w, 10):
            try:
                end_point = (int(j + flow[i, j, 0]), int(i + flow[i, j, 1]))
                cv2.arrowedLine(
                    flow_image,
                    (j, i),
                    end_point,
                    (0, 255, 0),  # 綠色箭頭
                    1,
                    tipLength=0.3,
                )
            except Exception as e:
                continue
    
    # 計算光流強度與角度
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if np.std(magnitude) < magnitude_std_thresh:
        print("Magnitude lower than threshold, return None")
        return None, None, None

    # 正規化光流強度並生成強度圖
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_img = np.uint8(magnitude_normalized)
    color_magnitude = cv2.applyColorMap(magnitude_img, cv2.COLORMAP_JET)
    
    foe_x, foe_y = calculate_foe(flow, magnitude)
    
    # cv2.circle(flow_image, (int(foe_x), int(foe_y)), 10, (0, 0, 255), -1)  # 紅色圓點

    z_values = []

    h, w = gray_frame1.shape
    for i in range(0, h, 10):
        for j in range(0, w, 10):
            try:
                dx, dy = flow[i, j]
                x_mid = j + dx / 2
                y_mid = i + dy / 2
                D_mid = calculate_distance(x_mid, y_mid, foe_x, foe_y)
                m = np.sqrt(flow[i, j, 0]**2 + flow[i, j, 1]**2) # 光流長度
                # print(m)
                z = 2.2 * D_mid / (m + 1e-6)  # 避免除零
                z_values.append(z)
            except Exception as e:
                z_values.append(0)

    print(f"Min z_value: {min(z_values)}, Max z_value: {max(z_values)}, Mean z_value: {np.mean(z_values)}")
    
    expected_length = len(range(0, h, 10)) * len(range(0, w, 10))
    print(f"z_values length: {len(z_values)}, expected length: {expected_length}")

    if len(z_values) != expected_length:
        z_values.extend([0] * (expected_length - len(z_values)))  # 填充至預期長度

    print("z_values", z_values)
    z_array = np.array(z_values).reshape(len(range(0, h, 10)), len(range(0, w, 10)))
    
    print("z_array", z_array)
    z_normalized = cv2.normalize(z_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("z_normalized", z_normalized)
    print(f"Before normalization: Min z_array: {np.min(z_normalized)}, Max z_array: {np.max(z_normalized)}")
    z_resized = cv2.resize(z_normalized, (w, h), interpolation=cv2.INTER_NEAREST)

    print(z_resized.shape[0], z_resized.shape[1])
    return flow_image, z_resized, color_magnitude

def calculate_foe(flow, magnitude):
    h, w = magnitude.shape
    x_sum, y_sum, count = 0, 0, 0

    for i in range(0, h, 10):
        for j in range(0, w, 10):
            if magnitude[i, j] > 5:
                fx, fy = flow[i, j]
                x_sum += j - fx
                y_sum += i - fy
                count += 1

    return (x_sum / count, y_sum / count) if count > 0 else (None, None)

if __name__ == "__main__":
    image1_path = "../image/test01.png"
    image2_path = "../image/test02.png"

    flow_image, z_gray, color_magnitude = calculate_optical_flow(image1_path, image2_path)

    if flow_image is not None and z_gray is not None:
        cv2.imshow("Optical Flow", flow_image)
        cv2.imshow("Magnitude Color Map", color_magnitude)
        plt.imshow(z_gray, cmap='gray')
        plt.title("Depth Map (Z) as Grayscale Image")
        plt.show()

    cv2.destroyAllWindows()
