import cv2
import numpy as np
import mediapipe as mp


class SquatAnalyzer:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )

    def calc_angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def analyze(self, video_path: str) -> dict:

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

        mp_draw = mp.solutions.drawing_utils

        counter = 0
        stage = "hore"
        correct_form = 0
        bad_form = 0
        drep_log = []
        rep_feedback = []

        min_knee_angle = 180
        min_trunk_angle = 180

        knee_buffer = []
        trunk_buffer = []
        head_buffer = []
        SMOOTH_WINDOW = 5

        down_frames = 0
        up_frames = 0

        feedback_text = ""
        feedback_timer = 0
        feedback_duration = int(fps)  # 1 sekunda

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            height, width, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.pose.process(img_rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                hip = np.array([lm[23].x * width, lm[23].y * height])
                knee = np.array([lm[25].x * width, lm[25].y * height])
                ankle = np.array([lm[27].x * width, lm[27].y * height])
                shoulder = np.array([lm[11].x * width, lm[11].y * height])
                ear = np.array([lm[7].x * width, lm[7].y * height])

                raw_knee = self.calc_angle(hip, knee, ankle)
                raw_trunk = self.calc_angle(shoulder, hip, knee)
                raw_head = self.calc_angle(ear, shoulder, hip)

                knee_buffer.append(raw_knee)
                trunk_buffer.append(raw_trunk)
                head_buffer.append(raw_head)

                if len(knee_buffer) > SMOOTH_WINDOW:
                    knee_buffer.pop(0)
                    trunk_buffer.pop(0)
                    head_buffer.pop(0)

                knee_angle = sum(knee_buffer) / len(knee_buffer)
                trunk_angle = sum(trunk_buffer) / len(trunk_buffer)
                head_angle = sum(head_buffer) / len(head_buffer)

                mp_draw.draw_landmarks(
                    img,
                    res.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                # Zobrazovanie uhlov - to by v apke nemalo ani byt VYMAZAT POTOM
                cv2.putText(img, f"Knee: {int(knee_angle)}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                cv2.putText(img, f"Trunk: {int(trunk_angle)}",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

                cv2.putText(img, f"Head: {int(head_angle)}",
                            (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

                # stage protection
                if knee_angle < 105:
                    down_frames += 1
                else:
                    down_frames = 0

                if knee_angle > 160:
                    up_frames += 1
                else:
                    up_frames = 0

                if down_frames > 3 and stage == "hore":
                    stage = "dole"
                    min_knee_angle = knee_angle
                    min_trunk_angle = trunk_angle

                if stage == "dole":
                    min_knee_angle = min(min_knee_angle, knee_angle)
                    min_trunk_angle = min(min_trunk_angle, trunk_angle)

                if up_frames > 3 and stage == "dole":
                    stage = "hore"
                    counter += 1

                    is_deep = min_knee_angle < 85
                    is_upright = min_trunk_angle > 35
                    is_head_aligned = head_angle > 145

                    if is_deep and is_upright and is_head_aligned:
                        correct_form += 1
                        rep_feedback.append({
                            "rep": counter,
                            "status": "correct",
                            "message": "Správne vykonané"
                        })
                    else:
                        bad_form += 1

                        reason = "Nešpecifikovaná chyba"
                        if not is_deep:
                            reason = "Plytký drep"
                        elif not is_upright:
                            reason = "Predklon trupu"
                        elif not is_head_aligned:
                            reason = "Hlava mimo osi"

                        rep_feedback.append({
                            "rep": counter,
                            "status": "incorrect",
                            "message": reason
                        })


                    feedback_timer = feedback_duration
                    min_knee_angle = 180
                    min_trunk_angle = 180

            # feedback display
            if feedback_timer > 0:
                cv2.putText(img, feedback_text,
                            (50,160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 3)
                feedback_timer -= 1

            cv2.putText(img, f"Reps: {counter}",
                        (width - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            out.write(img)

        cap.release()
        out.release()

        total = correct_form + bad_form
        accuracy = round(correct_form / total * 100, 2) if total > 0 else 0

        return {
            "exercise": "squat",
            "reps": total,
            "correct_reps": correct_form,
            "incorrect_reps": bad_form,
            "accuracy_percent": accuracy,
            "rep_feedback": rep_feedback,
            "output_video": "output.mp4"
        }

