import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class FlowCreateFoe:
    def __init__(self):
        self.magnitude_std_thresh = 0
        self.z_values = []
        self.sum_z_resized = None  # 累加 z_resized 的變量
        self.frame_count = 0       # 計算總幀數
        self.all_z_values = []     # 用來記錄所有 z 值
        self.bg_subtractor =  cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)  # 初始化 MOG2 背景建模器
        self.tracked_objects = {}  # Dictionary to store tracked objects

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
        if np.std(magnitude) < self.magnitude_std_thresh:
            return None, None
        return flow, magnitude

    def create_force_image(self, flow, magnitude, frame):
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_img = np.uint8(magnitude_normalized)
        color_magnitude = cv2.applyColorMap(magnitude_img, cv2.COLORMAP_JET)

        foe_x, foe_y = self.calculate_foe(flow, magnitude)
        if foe_x is not None and foe_y is not None:
            cv2.circle(color_magnitude, (int(foe_x), int(foe_y)), 10, (0, 0, 255), -1)  # Red circle
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
        self.z_values = []  # Reset z_values
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                try:
                    dx, dy = flow[i, j]
                    x_mid = j + dx / 2
                    y_mid = i + dy / 2
                    D_mid = self.calculate_distance(x_mid, y_mid, foe_x, foe_y)
                    m = np.sqrt(dx ** 2 + dy ** 2)  # Optical flow length
                    z = 2.2 * D_mid / (m + 1)  # Avoid division by zero
                    self.z_values.append(z)
                except Exception as e:
                    self.z_values.append(0)

        z_max = max(self.z_values)
        z_min = min(self.z_values)
        if z_max - z_min != 0:
            z_normalized = [(z - z_min) / (z_max - z_min) * 255 for z in self.z_values]
        else:
            z_normalized = [0 for _ in self.z_values]

        z_normalized = np.array(z_normalized, dtype=np.uint8)
        z_array = z_normalized.reshape(len(range(0, h, 10)), len(range(0, w, 10)))
        z_resized =  cv2.resize(z_array, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.sum_z_resized is None:
            self.sum_z_resized = np.zeros_like(z_resized, dtype=np.float32)

        self.sum_z_resized += z_resized
        self.frame_count += 1
        return z_resized
    
    def get_average_z_resized(self):
        return (self.sum_z_resized / self.frame_count).astype(np.uint8)

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

    def track_object(self, foe_x, foe_y, frame_counter):
        # Start tracking object and record its coordinates
        if foe_x is not None and foe_y is not None:
            object_id = len(self.tracked_objects) + 1
            if object_id not in self.tracked_objects:
                self.tracked_objects[object_id] = {
                    "frame_counter": frame_counter,
                    "appear_time": time.strftime('%H:%M:%S', time.gmtime(frame_counter / 30)),  # Assume 30 FPS
                    "disappear_time": None,
                    "circle_center": []
                }
            self.tracked_objects[object_id]["circle_center"].append([int(foe_x), int(foe_y)])
            # Update disappear time each frame
            self.tracked_objects[object_id]["disappear_time"] = time.strftime('%H:%M:%S', time.gmtime(frame_counter / 30))

    def save_tracking_results(self, filename='tracking_results.json'):
        import json
        with open(filename, 'w') as f:
            json.dump(self.tracked_objects, f, indent=4)

if __name__ == "__main__":
    flow_create_foe = FlowCreateFoe()
    cap = cv2.VideoCapture("../video/755776412.816110.mp4")
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        cap.release()
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prep_prev_frame, prep_frame = flow_create_foe.preprocess_image(prev_gray, gray)
        flow, magnitude = flow_create_foe.calculate_optical_flow(prep_prev_frame, prep_frame)
        if flow is not None and magnitude is not None:
            fg_mask = flow_create_foe.apply_background_subtraction(frame)
            fg_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

            color_magnitude, foe_x, foe_y = flow_create_foe.create_force_image(flow, magnitude, frame)
            flow_image = flow_create_foe.draw_flow(flow, frame.copy())

            z_resized = flow_create_foe.calculate_flow_length(frame.shape[0], frame.shape[1], flow, foe_x, foe_y)

            flow_create_foe.track_object(foe_x, foe_y, frame_counter)

            cv2.imshow("Optical Flow", flow_image)
            cv2.imshow("color_magnitude", color_magnitude)
            cv2.imshow("Z Resized", z_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        prev_gray = gray

    flow_create_foe.save_tracking_results('tracking_results.json')

    cap.release()
    cv2.destroyAllWindows()
