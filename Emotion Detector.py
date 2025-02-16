import cv2
from deepface import DeepFace

def generate_response(emotion):
    responses = {
        "angry": "I see you're feeling angry. Take a deep breath and relax.",
        "disgust": "You seem disgusted. Let's talk about what's bothering you.",
        "fear": "It looks like you're scared. Don't worry, I'm here to help.",
        "happy": "You're happy! That's wonderful to see!",
        "sad": "You look sad. Is there anything I can do to cheer you up?",
        "surprise": "You seem surprised! What's the good news?",
        "neutral": "You seem neutral. How can I assist you today?"
    }
    return responses.get(emotion, "I'm not sure how you're feeling. Can you tell me more?")

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            emotion = result[0]['dominant_emotion']

            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            response = generate_response(emotion)
            print(f"Bot: {response}")

        except Exception as e:
            print(f"Error: {e}")
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()