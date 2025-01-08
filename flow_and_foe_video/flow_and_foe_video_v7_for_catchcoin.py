import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
function 名稱修正，移除apply_background_subtraction
將calculate_foe_coordinate 修改，根據物件大小計算FOE座標
將資料各函式的輸入參數修改
"""
class FlowCreateFoe:
    def __init__(self):
        self.magnitude_std_thresh = 0
        self.step = 10
        self.z_values = []
        self.sum_z_resized = None  # 累加 z_resized 的變量
        self.frame_count = 0       # 計算總幀數
        self.all_z_values = []     # 用來記錄所有 z 值

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

        if np.std(magnitude) < self.magnitude_std_thresh:
            print("Magnitude lower than threshold, returning None")
            return None, None
        return flow, magnitude

    def create_optical_flow_heatmap(self, magnitude):
        # 正規化光流強度並生成強度圖
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_img = np.uint8(magnitude_normalized)
        color_flow = cv2.applyColorMap(magnitude_img, cv2.COLORMAP_JET)
        return color_flow

    def calculate_foe_coordinate(self, flow, region):
        x1, y1, x2, y2 = region  # 提取區域範圍
        points = []  # 存儲光流的起點
        directions = []  # 存儲光流的方向

        for y in range(y1, y2, self.step):
            for x in range(x1, x2, self.step):
                fx, fy = flow[y, x]
                if np.sqrt(fx**2 + fy**2) > 1:  # 過濾掉太小的光流
                    points.append([x, y])
                    directions.append([fx, fy])

        points = np.array(points)
        directions = np.array(directions)
        
        if len(points) == 0 or len(directions) == 0:
            print("No valid optical flow vectors found in region.")
            return None, None

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
            # print("foc", foc)
            return foc
        except np.linalg.LinAlgError:
            return None

    def mark_foe(self, frame, foc):
        if foc is not None:
            foc = tuple(map(int, foc))
            cv2.circle(frame, foc, 10, (0, 255, 255), -1)  # 在匯集點繪製紅色圓形
            cv2.putText(
                frame,
                "FOE", 
                (foc[0] + 10, foc[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 1
            )
        return frame

    def draw_flow(self, flow, frame, fore_original_region_x, fore_original_region_y, region):
        x1, y1, x2, y2 = region  # 提取區域範圍
        for y in range(y1, y2, self.step):
            for x in range(x1, x2, self.step):
                try:
                    fx, fy = flow[y, x]
                    start_point = (int(fore_original_region_x + x), int(fore_original_region_y + y))
                    end_point = (int(fore_original_region_x + x + fx), int(fore_original_region_y + y + fy))
                    magnitude = np.sqrt(fx**2 + fy**2)
                    color = (int(min(magnitude * 10, 255)), 255 - int(min(magnitude * 10, 255)), 0)
                    cv2.arrowedLine(frame, start_point, end_point, color, 1, tipLength=0.3)
                except Exception as e:
                    continue
        return frame

    def calculate_distance(self, x, y, foe_x, foe_y):
        return np.sqrt((x - foe_x) ** 2 + (y - foe_y) ** 2)

    def create_depth_image(self, flow, foe_x, foe_y, region):
        self.z_values = []  # 重置 z_values
        x1, y1, x2, y2 = region  # 提取區域範圍
        
        for i in range(y1, y2, self.step):
            for j in range(x1, x2, self.step):
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
        z_array = z_normalized.reshape(len(range(y1, y2, 10)), len(range(x1, x2, 10)))

        # 調整尺寸
        z_resized =  cv2.resize(z_array, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        # 累加 z_resized
        if self.sum_z_resized is None:
            self.sum_z_resized = np.zeros_like(z_resized, dtype=np.float32)

        # self.sum_z_resized += z_resized
        self.frame_count += 1

        # Add current frame's z_values to the all_z_values list
        self.all_z_values.extend(self.z_values)
        
        z_resized = 255 - z_resized

        return z_resized

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pre_img = cv2.equalizeHist(gray)
    return pre_img

def save_video(filename, fps=30, w=720, h=480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    return out

if __name__ == "__main__":
    paused = False
    flow_create_foe = FlowCreateFoe()

    cap = cv2.VideoCapture("../catchcoin/background.mp4")  #./video/755776412.816110.mp4 # ./video/XR21-14-15_XQ17-03-04_out.mp4
    ret, prev_frame = cap.read()
    h, w = prev_frame.shape[:2]
    frame_region = (0, 0, w, h)
    out_optical_flow = save_video("optical_flow_background.mp4", fps=30)
    out_optical_flow_heatmap = save_video("optical_flow_heatmap_background.mp4", fps=30)
    out_depth_image = save_video("depth_image.mp4", fps=30)

    if not ret:
        print("Error reading video file")
        cap.release()
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 前處理
        prep_prev_frame = preprocess_image(prev_frame)
        prep_frame = preprocess_image(frame)

        # 計算光流(Farneback)
        flow, magnitude = flow_create_foe.calculate_optical_flow(prep_prev_frame, prep_frame)

        if flow is not None and magnitude is not None:
            # 產生光流強度圖(OpticalFlowHeatmap)
            optical_flow_heatmap = flow_create_foe.create_optical_flow_heatmap(magnitude)
            # 計算FOE座標
            foe_x, foe_y = flow_create_foe.calculate_foe_coordinate(flow, frame_region)
            if foe_x is None or foe_y is None:
                print("No FOE found.")
                continue
            # 標記FOE在畫面上
            flow_create_foe.mark_foe(frame, (foe_x, foe_y))
            # 標記光流流向在畫面上
            flow_image = flow_create_foe.draw_flow(flow, frame.copy(), 0, 0, frame_region)
            # 產生深度圖
            depth_image = flow_create_foe.create_depth_image(flow, foe_x, foe_y, frame_region)
            
            cv2.imshow("optical_flow", flow_image)
            cv2.imshow("optical_flow_heatmap", optical_flow_heatmap)
            cv2.imshow("depth_image", depth_image)  # 顯示 z_resized
            
            
            out_optical_flow.write(flow_image)
            out_optical_flow_heatmap.write(optical_flow_heatmap)
            
            # out_depth_image.write(depth_image)
            
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
            
        prev_frame = frame

    cap.release()
    out_optical_flow.release()
    out_optical_flow_heatmap.release()
    # out_depth_image.release()
    cv2.destroyAllWindows()
