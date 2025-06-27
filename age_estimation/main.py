import cv2
from deepface import DeepFace


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
eye_glasses_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def check_face_quality(face_gray):
    """Return (ok, message) depending on glasses/mask/grimace detection."""
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)
    glasses = eye_glasses_cascade.detectMultiScale(face_gray, 1.1, 4)
    if len(eyes) == 0 and len(glasses) > 0:
        return False, "Glasses detected"

    smiles = smile_cascade.detectMultiScale(face_gray, 1.7, 20)
    if len(smiles) == 0:
        return False, "Mask or occlusion detected"
    for (_, _, _, h) in smiles:
        if h > face_gray.shape[0] * 0.3:
            return False, "Grimace detected"
    return True, ""


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            message = "No face detected"
            color = (0, 0, 255)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_gray = gray[y : y + h, x : x + w]
                ok, msg = check_face_quality(face_gray)
                if ok:
                    result = DeepFace.analyze(
                        frame[y : y + h, x : x + w],
                        actions=["age"],
                        enforce_detection=False,
                    )
                    age = result.get("age", "N/A")
                    message = f"Age: {age}"
                    color = (0, 255, 0)
                else:
                    message = msg

            cv2.putText(
                frame,
                message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Age Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
