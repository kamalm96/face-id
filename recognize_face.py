import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import argparse

EMBEDDINGS_FILE = "known_embeddings.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.70
print(f"Running on device: {DEVICE}")

mtcnn = MTCNN(
    keep_all=True, device=DEVICE, min_face_size=20, thresholds=[0.6, 0.7, 0.7]
)
resnet = InceptionResnetV1(pretrained="vggface2", classify=False, device=DEVICE).eval()


def load_known_faces():
    """Loads known face embeddings from the EMBEDDINGS_FILE."""
    if os.path.exists(EMBEDDINGS_FILE):
        known_embeddings_data = torch.load(EMBEDDINGS_FILE, map_location=DEVICE)
        print(f"Loaded known embeddings for: {list(known_embeddings_data.keys())}")
        return known_embeddings_data
    else:
        print(
            f"Error: Embeddings file '{EMBEDDINGS_FILE}' not found. Please run enroll_faces.py first."
        )
        return {}


def recognize_faces(image_source, known_embeddings_data):

    if not known_embeddings_data:
        print("No known faces to compare against. Exiting.")
        return

    if isinstance(image_source, str) and os.path.isfile(image_source):
        img_pil = Image.open(image_source).convert("RGB")
        process_frame(img_pil, known_embeddings_data, display=True)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
    elif image_source == 0:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Webcam started. Press 'q' to quit.")
        while True:
            ret, frame_cv = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))

            processed_frame_pil = process_frame(
                frame_pil, known_embeddings_data, display=False
            )

            processed_frame_cv = cv2.cvtColor(
                np.array(processed_frame_pil), cv2.COLOR_RGB2BGR
            )
            cv2.imshow("Face Recognition - Press Q to Quit", processed_frame_cv)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print(
            f"Error: Invalid image source '{image_source}'. Provide a file path or 0 for webcam."
        )


def process_frame(img_pil, known_embeddings_data, display=True):
    boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)

    img_cropped_list = mtcnn(img_pil, save_path=None)

    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    if img_cropped_list is not None:
        for i, face_tensor in enumerate(img_cropped_list):
            face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
            current_embedding = resnet(face_tensor).detach().cpu()

            best_match_name = "Unknown"
            best_similarity = -1

            for name, known_emb in known_embeddings_data.items():
                similarity = torch.nn.functional.cosine_similarity(
                    current_embedding, known_emb.to(DEVICE)
                ).item()

                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity > SIMILARITY_THRESHOLD:
                        best_match_name = name
                    else:
                        best_match_name = "Unknown"

            box = boxes[i]
            draw.rectangle(box.tolist(), outline="lime", width=2)
            text_position = (
                box[0],
                box[1] - 20 if box[1] - 20 > 0 else box[1] + 5,
            )
            draw.text(
                text_position,
                f"{best_match_name} ({best_similarity:.2f})",
                fill="lime",
                font=font,
            )
    if display and not (isinstance(image_source, int) and image_source == 0):
        img_pil.show()

    return img_pil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument(
        "--image_path",
        type=str,
        default="0",
        help="Path to the image file for recognition, or '0' for webcam.",
    )
    args = parser.parse_args()

    known_faces_data = load_known_faces()

    image_source_arg = 0 if args.image_path == "0" else args.image_path
    recognize_faces(image_source_arg, known_faces_data)
