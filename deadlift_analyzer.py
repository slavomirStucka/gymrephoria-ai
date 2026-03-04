import cv2
import numpy as np
import mediapipe as mp


class EMA:
    def __init__(self, alpha=0.3, init=None):
        self.alpha = alpha
        self.val = init

    def update(self, x):
        if self.val is None:
            self.val = x
        else:
            self.val = (1 - self.alpha) * self.val + self.alpha * x
        return self.val


class DeadliftAnalyzer:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def angle_between(self, v1, v2):
        EPS = 1e-6
        v1 = v1 / (np.linalg.norm(v1) + EPS)
        v2 = v2 / (np.linalg.norm(v2) + EPS)
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    def joint_angle(self, a, b, c):
        return self.angle_between(a - b, c - b)

    def analyze(self, video_path: str) -> dict:

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

        mp_draw = mp.solutions.drawing_utils

        counter = 0
        stage = "hore"
        correct_form = 0
        bad_form = 0
        deadlift_log = []
        rep_feedback = []

        T_UP = 20.0
        T_DOWN = 60.0
        HYS = 5.0

        max_trunk_flexion = 0.0
        ema_trunk = EMA(alpha=0.3)

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.pose.process(img_rgb)

            if res.pose_landmarks:

                lm2d = res.pose_landmarks.landmark
                world_lm = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

                mp_draw.draw_landmarks(img, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                side = "L" if lm2d[23].visibility >= lm2d[24].visibility else "R"

                S = self.mp_pose.PoseLandmark.LEFT_SHOULDER if side == "L" else self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                H = self.mp_pose.PoseLandmark.LEFT_HIP if side == "L" else self.mp_pose.PoseLandmark.RIGHT_HIP

                shoulder = np.array([lm2d[S].x * width, lm2d[S].y * height])
                hip = np.array([lm2d[H].x * width, lm2d[H].y * height])

                if world_lm:
                    shoulder3 = np.array([world_lm[S].x, world_lm[S].y, world_lm[S].z])
                    hip3 = np.array([world_lm[H].x, world_lm[H].y, world_lm[H].z])
                    v_trunk = shoulder3 - hip3
                    v_up = np.array([0.0, -1.0, 0.0])
                    t_angle_raw = self.angle_between(v_trunk, v_up)
                else:
                    v_trunk = shoulder - hip
                    v_up = np.array([0.0, -1.0])
                    t_angle_raw = self.angle_between(np.append(v_trunk,0), np.append(v_up,0))

                t_angle = ema_trunk.update(t_angle_raw)

                # -------- REP LOGIKA --------
                if stage == "hore":
                    if t_angle > (T_DOWN + HYS):
                        stage = "dole"
                        max_trunk_flexion = t_angle
                else:
                    max_trunk_flexion = max(max_trunk_flexion, t_angle)

                    if t_angle < (T_UP - HYS):
                        stage = "hore"
                        counter += 1

                        is_back_ok = max_trunk_flexion >= 70.0

                        if is_back_ok:
                            correct_form += 1
                            rep_feedback.append({
                                "rep": counter,
                                "status": "correct",
                                "message": "Správny mŕtvy ťah"
                            })
                        else:
                            bad_form += 1
                            reason = "ohnutý chrbát / nadmerný predklon"
                            deadlift_log.append({"id": counter, "reason": reason})
                            rep_feedback.append({
                                "rep": counter,
                                "status": "incorrect",
                                "message": reason
                            })

                cv2.putText(img, f"Reps: {counter}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            out.write(img)

        cap.release()
        out.release()

        total = correct_form + bad_form
        accuracy = round(correct_form / total * 100, 2) if total > 0 else 0

        return {
            "exercise": "deadlift",
            "reps": total,
            "correct_reps": correct_form,
            "incorrect_reps": bad_form,
            "accuracy_percent": accuracy,
            "rep_feedback": rep_feedback,
            "output_video": "output.mp4"
        }