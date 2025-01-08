import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
將每前後幀做深度圖，加總各像素並除以總幀數
"""

class FlowCreateFoe:
    def __init__(self):
        self.magnitude_std_thresh = 5
        self.z_values = []
        self.sum_z_resized = None  # 累加 z_resized 的變量
        self.frame_count = 0       # 計算總幀數

    def calculate_distance(self, x, y, foe_x, foe_y):
        return np.sqrt((x - foe_x) ** 2 + (y - foe_y) ** 2)

    def calculate_optical_flow(self, gray_frame1, gray_frame2):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frame1,
            gray_frame2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=50,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        if np.std(magnitude) < self.magnitude_std_thresh:
            print("Magnitude lower than threshold, returning None")
            return None, None
        return flow, magnitude

    def create_force_image(self, flow, magnitude, frame):
        # 正規化光流強度並生成強度圖
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_img = np.uint8(magnitude_normalized)
        color_magnitude = cv2.applyColorMap(magnitude_img, cv2.COLORMAP_JET)

        foe_x, foe_y = self.calculate_foe(flow, magnitude)

        if foe_x is not None and foe_y is not None:
            cv2.circle(color_magnitude, (int(foe_x), int(foe_y)), 10, (0, 0, 255), -1)  # 红色圆点
        return color_magnitude, foe_x, foe_y

    def draw_flow(self, flow, frame):
        h, w, _ = frame.shape
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                try:
                    end_point = (int(j + flow[i, j, 0]), int(i + flow[i, j, 1]))
                    cv2.arrowedLine(
                        frame,
                        (j, i),
                        end_point,
                        (0, 255, 0),   # 綠色箭頭
                        1,
                        tipLength=0.3,
                    )
                except Exception as e:
                    continue
        return frame

    def calculate_flow_length(self, h, w, flow, foe_x, foe_y):
        self.z_values = []  # 重置 z_values
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                try:
                    dx, dy = flow[i, j]
                    x_mid = j + dx / 2
                    y_mid = i + dy / 2
                    D_mid = self.calculate_distance(x_mid, y_mid, foe_x, foe_y)
                    m = np.sqrt(dx ** 2 + dy ** 2)  # 光流長度
                    z = 2.2 * D_mid / (m + 1)  # 避免除零
                    self.z_values.append(z)
                except Exception as e:
                    self.z_values.append(0)

        z_array = np.array(self.z_values).reshape(len(range(0, h, 10)), len(range(0, w, 10)))
        z_normalized = cv2.normalize(z_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        z_resized = cv2.resize(z_normalized, (w, h), interpolation=cv2.INTER_NEAREST)

        # 累加 z_resized
        if self.sum_z_resized is None:
            self.sum_z_resized = np.zeros_like(z_resized, dtype=np.float32)

        self.sum_z_resized += z_resized
        self.frame_count += 1

        return z_resized
    
    def get_average_z_resized(self):
        # if self.frame_count == 0:
        #     return None
        print(self.sum_z_resized)
        average_z_resized = (self.sum_z_resized / self.frame_count).astype(np.uint8)
        return average_z_resized

    def calculate_foe(self, flow, magnitude):
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
    flow_create_foe = FlowCreateFoe()
    cap = cv2.VideoCapture("./kitti/kitti.mp4")  
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        cap.release()
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow, magnitude = flow_create_foe.calculate_optical_flow(prev_gray, gray)
        if flow is not None and magnitude is not None:
            color_magnitude, foe_x, foe_y = flow_create_foe.create_force_image(flow, magnitude, frame)
            flow_image = flow_create_foe.draw_flow(flow, frame.copy())

            # 呼叫 calculate_flow_length，並顯示結果
            h, w = gray.shape
            z_resized = flow_create_foe.calculate_flow_length(h, w, flow, foe_x, foe_y)

            cv2.imshow("Optical Flow", flow_image)
            cv2.imshow("Magnitude", color_magnitude)
            cv2.imshow("Z Resized", z_resized)  # 顯示 z_resized

            if cv2.waitKey(30) & 0xFF == 27:  # 按 ESC 鍵退出
                break

        prev_gray = gray


    cap.release()
    cv2.destroyAllWindows()
    
    average_z_resized = flow_create_foe.get_average_z_resized()
    print(average_z_resized)
    if average_z_resized is not None:
        cv2.imwrite('average_z_resized.png', average_z_resized)
        print("Saved average_z_resized.png")
    else:
        print("No frames processed.")
