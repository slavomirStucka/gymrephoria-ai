import cv2
import numpy as np
import mediapipe as mp


class PullupAnalyzer:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze(self, video_path: str) -> dict:

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

        mp_draw = mp.solutions.drawing_utils

        def get_xy(lm, idx):
            return np.array([lm[idx].x * width, lm[idx].y * height], dtype=np.float32)

        def visible(lm, idx, thr=0.5):
            return lm[idx].visibility >= thr

        def y_on_line_at_x(p1, p2, x):
            if abs(p2[0] - p1[0]) < 1e-3:
                return float((p1[1] + p2[1]) * 0.5)
            t = (x - p1[0]) / (p2[0] - p1[0])
            return float(p1[1] + t * (p2[1] - p1[1]))

        def calc_angle(a, b, c):
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosang = float(np.dot(ba, bc) / denom)
            return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

        def elbow_angles_deg(lm):
            L = self.mp_pose.PoseLandmark
            angL = angR = None
            if visible(lm, L.LEFT_SHOULDER, 0.3) and visible(lm, L.LEFT_ELBOW, 0.3) and visible(lm, L.LEFT_WRIST, 0.3):
                s = get_xy(lm, L.LEFT_SHOULDER)
                e = get_xy(lm, L.LEFT_ELBOW)
                w = get_xy(lm, L.LEFT_WRIST)
                angL = calc_angle(s, e, w)

            if visible(lm, L.RIGHT_SHOULDER, 0.3) and visible(lm, L.RIGHT_ELBOW, 0.3) and visible(lm, L.RIGHT_WRIST, 0.3):
                s = get_xy(lm, L.RIGHT_SHOULDER)
                e = get_xy(lm, L.RIGHT_ELBOW)
                w = get_xy(lm, L.RIGHT_WRIST)
                angR = calc_angle(s, e, w)

            return angL, angR

        counter = 0
        stage = "dole"
        correct_form = 0
        bad_form = 0
        rep_feedback = []

        wristL_cache = None
        wristR_cache = None

        HYS_PX = int(0.02 * height)
        ELBOW_EXT_THR = 155.0

        rep_max_elbow_ext = 0.0
        pending_bottom_eval = False
        frame_idx = 0
        bottom_start_frame = 0
        prev_ext_for_deriv = None
        stable_frames = 0
        ELBOW_DERIV_EPS = 1.2
        BOTTOM_SETTLE_FRAMES = int(0.75 * fps)
        STABLE_FRAMES_REQ = 10

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            frame_idx += 1

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.pose.process(img_rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                mp_draw.draw_landmarks(img, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                LWI = self.mp_pose.PoseLandmark.LEFT_WRIST
                RWI = self.mp_pose.PoseLandmark.RIGHT_WRIST
                NOSE = self.mp_pose.PoseLandmark.NOSE

                if visible(lm, LWI): wristL_cache = get_xy(lm, LWI)
                if visible(lm, RWI): wristR_cache = get_xy(lm, RWI)

                if wristL_cache is not None and wristR_cache is not None:

                    pL = wristL_cache
                    pR = wristR_cache

                    if visible(lm, NOSE, 0.2):

                        nose = get_xy(lm, NOSE)
                        bar_y_at_nose = y_on_line_at_x(pL, pR, nose[0])

                        is_above = (nose[1] <= bar_y_at_nose - HYS_PX)
                        is_below = (nose[1] >= bar_y_at_nose + HYS_PX)

                        angL, angR = elbow_angles_deg(lm)

                        ext_candidate = None
                        if angL is not None and angR is not None:
                            ext_candidate = min(angL, angR)
                        elif angL is not None:
                            ext_candidate = angL
                        elif angR is not None:
                            ext_candidate = angR

                        if ext_candidate is not None:
                            rep_max_elbow_ext = max(rep_max_elbow_ext, ext_candidate)

                        if stage == "dole":
                            if is_above:
                                stage = "hore"
                                rep_max_elbow_ext = 0.0
                                pending_bottom_eval = False
                                prev_ext_for_deriv = None
                                stable_frames = 0
                        else:
                            if is_below:
                                stage = "dole"
                                pending_bottom_eval = True
                                bottom_start_frame = frame_idx
                                prev_ext_for_deriv = ext_candidate
                                stable_frames = 0

                        if pending_bottom_eval:

                            if ext_candidate is not None and prev_ext_for_deriv is not None:
                                d = ext_candidate - prev_ext_for_deriv
                                if d > ELBOW_DERIV_EPS:
                                    stable_frames = 0
                                else:
                                    stable_frames += 1
                                prev_ext_for_deriv = ext_candidate

                            time_ok = (frame_idx - bottom_start_frame) >= BOTTOM_SETTLE_FRAMES
                            stable_ok = stable_frames >= STABLE_FRAMES_REQ

                            if time_ok or stable_ok:

                                counter += 1
                                full_extension = rep_max_elbow_ext >= ELBOW_EXT_THR

                                if full_extension:
                                    correct_form += 1
                                    rep_feedback.append({
                                        "rep": counter,
                                        "status": "correct",
                                        "message": "Plná extenzia – správne"
                                    })
                                else:
                                    bad_form += 1
                                    rep_feedback.append({
                                        "rep": counter,
                                        "status": "incorrect",
                                        "message": "Chýba plná extenzia"
                                    })

                                pending_bottom_eval = False

                cv2.putText(img, f"Reps: {counter}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            out.write(img)

        cap.release()
        out.release()

        total = correct_form + bad_form
        accuracy = round(correct_form / total * 100, 2) if total > 0 else 0

        return {
            "exercise": "pullups",
            "reps": total,
            "correct_reps": correct_form,
            "incorrect_reps": bad_form,
            "accuracy_percent": accuracy,
            "rep_feedback": rep_feedback,
            "output_video": "output.mp4"
        }