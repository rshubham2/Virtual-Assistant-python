import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class Vision:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Use CUDA if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        self.id2label = self.model.config.id2label
        self.focus_window_size = (300, 300)
        self.detection_interval = 5  # Detect every 5 frames for a balance of performance and responsiveness

    @torch.no_grad()
    def detect_object(self, frame):
        # Convert frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        inputs = self.image_processor(images=frame_rgb, return_tensors="pt").to(self.device)

        # Run inference
        outputs = self.model(**inputs)

        # Post-process the results
        target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[
            0]

        if len(results["scores"]) > 0:
            best_detection_index = results["scores"].argmax()
            label = self.id2label[results["labels"][best_detection_index].item()]
            confidence = results["scores"][best_detection_index].item()
            return {"label": label, "confidence": confidence}
        return None

    def create_focus_window(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        x1 = center_x - self.focus_window_size[0] // 2
        y1 = center_y - self.focus_window_size[1] // 2
        x2 = x1 + self.focus_window_size[0]
        y2 = y1 + self.focus_window_size[1]

        # Create a mask for the focus window
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Apply the mask
        blurred = cv2.GaussianBlur(frame, (31, 31), 0)
        result = np.where(mask[:, :, None] == 255, frame, blurred)

        # Draw rectangle around focus window
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result, frame[y1:y2, x1:x2]

    def run_detection(self, callback):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        last_detection = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_with_focus, focus_area = self.create_focus_window(frame)

            if frame_count % self.detection_interval == 0:
                detection = self.detect_object(focus_area)
                if detection:
                    last_detection = detection
                    callback([detection])

            if last_detection:
                label = f"{last_detection['label']}: {last_detection['confidence']:.2f}"
                cv2.putText(frame_with_focus, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Object Detection', frame_with_focus)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()