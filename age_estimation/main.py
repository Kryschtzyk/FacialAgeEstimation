import cv2
from deepface import DeepFace


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

            result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
            age = result.get('age', 'N/A')

            cv2.putText(
                frame,
                f"Age: {age}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
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
