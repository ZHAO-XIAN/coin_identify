import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
修正FOE，計算光流交會點 == FOE
"""
class FlowCreateFoe:
    def __init__(self):
        self.magnitude_std_thresh = 0
        self.z_values = []
        self.sum_z_resized = None  # 累加 z_resized 的變量
        self.frame_count = 0       # 計算總幀數
        self.all_z_values = []     # 用來記錄所有 z 值
        self.bg_subtractor =  cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)  # 初始化 MOG2 背景建模器

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
            poly_n=9,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (7, 7), 0)
        flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (7, 7), 0)
        magnitude, angle_matrix = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        # print("np.std(magnitude)", np.std(magnitude))
        if np.std(magnitude) < self.magnitude_std_thresh:
            print("Magnitude lower than threshold, returning None")
            return None, None
        return flow, magnitude

    def create_force_image(self, flow, magnitude, frame):
        # 正規化光流強度並生成強度圖
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_img = np.uint8(magnitude_normalized)
        color_magnitude = cv2.applyColorMap(magnitude_img, cv2.COLORMAP_JET)

        foe_x, foe_y = self.find_focus_of_expansion(flow, frame)

        if foe_x is not None and foe_y is not None:
            cv2.circle(color_magnitude, (int(foe_x), int(foe_y)), 10, (0, 0, 255), -1)  # 红色圆点
        return color_magnitude, foe_x, foe_y

    def draw_flow(self, flow, frame, step=10):
        h, w = frame.shape[:2]
        for y in range(0, h, step):
            for x in range(0, w, step):
                try:
                    fx, fy = flow[y, x]
                    end_point = (int(x + fx), int(y + fy))
                    magnitude = np.sqrt(fx**2 + fy**2)
                    color = (int(min(magnitude * 10, 255)), 255 - int(min(magnitude * 10, 255)), 0)
                    cv2.arrowedLine(frame, (x, y), end_point, color, 1, tipLength=0.3)
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

        # 計算 z_values 的最大值和最小值
        z_max = max(self.z_values)
        z_min = min(self.z_values)
        # 避免除以 0 的情況並對 z_values 進行正規化
        if z_max - z_min != 0:
            z_normalized = [(z - z_min) / (z_max - z_min) * 255 for z in self.z_values]
        else:
            z_normalized = [0 for _ in self.z_values]

        # 轉換為 numpy 陣列並轉換為 uint8
        z_normalized = np.array(z_normalized, dtype=np.uint8)
        z_array = z_normalized.reshape(len(range(0, h, 10)), len(range(0, w, 10)))

        # 調整尺寸
        z_resized =  cv2.resize(z_array, (w, h), interpolation=cv2.INTER_LINEAR)

        # 累加 z_resized
        if self.sum_z_resized is None:
            self.sum_z_resized = np.zeros_like(z_resized, dtype=np.float32)

        self.sum_z_resized += z_resized
        self.frame_count += 1

        # Add current frame's z_values to the all_z_values list
        self.all_z_values.extend(self.z_values)
        
        # z_resized = 255 - z_resized

        return z_resized
    
    def get_average_z_resized(self):
        average_z_resized = (self.sum_z_resized / self.frame_count).astype(np.uint8)
        return average_z_resized

    def find_focus_of_expansion(self, flow, frame):
        step=10
        h, w = frame.shape[:2]
        points = []  # 存儲光流的起點
        directions = []  # 存儲光流的方向

        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                if np.sqrt(fx**2 + fy**2) > 1:  # 過濾掉太小的光流
                    points.append([x, y])
                    directions.append([fx, fy])

        points = np.array(points)
        directions = np.array(directions)

        # 構建矩陣 A 和 b
        A = np.zeros((len(points), 2))
        b = np.zeros(len(points))

        for i, (point, direction) in enumerate(zip(points, directions)):
            px, py = point
            dx, dy = direction
            A[i] = [-dy, dx]
            b[i] = dx * py - dy * px

        # 解最小二乘問題
        try:
            foc, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            print("foc", foc)
            return foc
        except np.linalg.LinAlgError:
            return None

    def draw_focus(self, frame, foc):
        if foc is not None:
            foc = tuple(map(int, foc))
            cv2.circle(frame, foc, 10, (0, 255, 255), -1)  # 在匯集點繪製紅色圓形
            cv2.putText(frame, "FOE", (foc[0] + 10, foc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def plot_z_values(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.all_z_values, bins=50, color='blue', edgecolor='black')
        plt.title("Distribution of Z values")
        plt.xlabel("Z value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def apply_background_subtraction(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.dilate(fg_mask, None, iterations=5)
        return fg_mask

    def preprocess_image(self, prev_gray, gray):
        prev_frame = cv2.equalizeHist(prev_gray)
        frame = cv2.equalizeHist(gray)
        
        return prev_frame, frame
        

if __name__ == "__main__":
    paused = False
    flow_create_foe = FlowCreateFoe()
    cap = cv2.VideoCapture("../video/kitti.mp4")  #./video/755776412.816110.mp4 # ./video/XR21-14-15_XQ17-03-04_out.mp4
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        cap.release()
        exit()
    print(prev_frame.shape[0], prev_frame.shape[1])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # prev_gray = cv2.resize(prev_gray, (360, 640), interpolation=cv2.INTER_AREA)
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA)s
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prep_prev_frame, prep_frame = flow_create_foe.preprocess_image(prev_gray, gray)
        flow, magnitude = flow_create_foe.calculate_optical_flow(prep_prev_frame, prep_frame)

        if flow is not None and magnitude is not None:
            # Apply background subtraction to focus on moving objects
            fg_mask = flow_create_foe.apply_background_subtraction(frame)
            fg_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

            color_magnitude, foe_x, foe_y = flow_create_foe.create_force_image(flow, magnitude, frame)
            flow_create_foe.draw_focus(frame, (foe_x, foe_y))
            flow_image = flow_create_foe.draw_flow(flow, frame.copy())

            # 呼叫 calculate_flow_length，並顯示結果
            h, w = gray.shape
            z_resized = flow_create_foe.calculate_flow_length(h, w, flow, foe_x, foe_y)

            cv2.imshow("Optical Flow", flow_image)
            cv2.imshow("color_magnitude", color_magnitude)
            cv2.imshow("Z Resized", z_resized)  # 顯示 z_resized
            # cv2.imshow('fg_frame', fg_frame)

            

            # 檢測按鍵，按 'q' 退出，按 'p' 暫停/繼續
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 'q' 退出
                break
            elif key == ord('p'):  # 按 'p' 暫停/繼續
                paused = not paused
                while paused:
                    key_pause = cv2.waitKey(1) & 0xFF
                    if key_pause == ord('p'):  # 再次按 'p' 以繼續
                        paused = False

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()
