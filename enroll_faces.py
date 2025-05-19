import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import glob
import numpy as np

KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_FILE = "known_embeddings.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")


mtcnn = MTCNN(image_size=160, margin=10, post_process=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2", classify=False, device=DEVICE).eval()


def enroll_faces():
    known_embeddings = {}

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"Processing images for: {person_name}")
        person_embeddings_list = []

        image_files = (
            glob.glob(os.path.join(person_dir, "*.jpg"))
            + glob.glob(os.path.join(person_dir, "*.jpeg"))
            + glob.glob(os.path.join(person_dir, "*.png"))
        )

        if not image_files:
            print(f"  No images found for {person_name}. Skipping.")
            continue

        for image_path in image_files:
            try:
                print(f"  Processing image: {os.path.basename(image_path)}")
                img = Image.open(image_path).convert("RGB")
                img_cropped_list = mtcnn(img, save_path=None)

                if img_cropped_list is not None:
                    face_tensor = (
                        img_cropped_list[0]
                        if isinstance(img_cropped_list, list)
                        else img_cropped_list
                    )
                    face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
                    embedding = resnet(face_tensor).detach().cpu()
                    person_embeddings_list.append(embedding)
                else:
                    print(f"    No face detected in {os.path.basename(image_path)}")
            except Exception as e:
                print(f"    Error processing {os.path.basename(image_path)}: {e}")

        if person_embeddings_list:
            all_embeddings_tensor = torch.cat(person_embeddings_list, dim=0)
            average_embedding = torch.mean(all_embeddings_tensor, dim=0, keepdim=True)
            known_embeddings[person_name] = average_embedding
            print(
                f"  Enrolled {person_name} with {len(person_embeddings_list)} images."
            )
        else:
            print(f"  Could not generate any embeddings for {person_name}.")

    torch.save(known_embeddings, EMBEDDINGS_FILE)
    print(f"\nEnrollment complete. Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Enrolled individuals: {list(known_embeddings.keys())}")


if __name__ == "__main__":
    enroll_faces()
