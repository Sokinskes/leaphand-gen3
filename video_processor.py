import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
from scipy.signal import savgol_filter


def extract_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    hand_trajectories = []  # 时序手部姿态 (序列, 63维: 21*3)
    object_poses = []  # 时序物体姿态 (序列, 3维: x,y,角度)
    point_cloud_seq = []  # 时序点云 (序列, 6144维)
    tactile_sim_seq = []  # 时序模拟触觉 (序列, 100维)
    frame_idx = 0
    with mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as mp_hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"[DEBUG] {video_path}: Processed {frame_idx} frames...")
            hand_joints = np.zeros(63)
            # 手部姿态
            results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                hand_joints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                hand_trajectories.append(savgol_filter(hand_joints, 5, 2))  # 滤波
            else:
                hand_trajectories.append(hand_joints)
            # 物体姿态 (假设HSV检测圆柱)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))  # 调整颜色范围
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                angle = 0.0
                if len(cnt) >= 5:
                    try:
                        angle = cv2.fitEllipse(cnt)[2]
                    except Exception:
                        angle = 0.0
                object_poses.append(np.array([x, y, angle]))
            else:
                object_poses.append(np.zeros(3))
            # 点云 (模拟深度)
            depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0 * 0.5  # 模拟深度图
            color_o3d = o3d.geometry.Image(frame)
            depth_o3d = o3d.geometry.Image((depth * 255).astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=255.0/0.5, convert_rgb_to_intensity=False
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
            )
            pcd.estimate_normals()
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            pc = np.concatenate([points.flatten(), normals.flatten()])
            if pc.size < 6144:
                pc = np.pad(pc, (0, 6144 - pc.size))
            else:
                pc = pc[:6144]
            point_cloud_seq.append(pc)
            # 模拟触觉 (从姿态推断力)
            tactile = np.random.randn(100) * 0.1 + np.linalg.norm(hand_joints[:3])  # 基于拇指速度
            tactile_sim_seq.append(tactile)
    cap.release()
    return hand_trajectories, object_poses, point_cloud_seq, tactile_sim_seq