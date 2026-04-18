import cv2 as cv
import torch
from torchvision import transforms
import numpy as np
import torchvision.models as models
import torch.nn as nn
from collections import deque

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
model = models.efficientnet_b0(weights=None) 
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, len(class_names))
    )
)

state_dict = torch.load('brain_tumor_efficientnet_model.pth', map_location=torch.device(device))
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

BUFFER_SIZE = 15 
prediction_history = deque(maxlen=BUFFER_SIZE)
stable_label = "Scanning..."
stable_conf = 0.0

def get_perspective_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype="float32")
    m = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(image, m, (max_width, max_height))

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    edged = cv.Canny(blurred, 50, 150)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    paper_contour = None
    if contours:
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
        for c in contours:
            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.03 * perimeter, True)
            if len(approx) == 4 and cv.contourArea(c) > (frame.shape[0] * frame.shape[1] * 0.1):
                paper_contour = approx
                break

    if paper_contour is not None:
        pts = paper_contour.reshape(4, 2)
        cv.drawContours(frame, [paper_contour], -1, (0, 255, 0), 2)
        for (x, y) in pts:
            cv.circle(frame, (x, y), 8, (0, 255, 0), cv.FILLED)

        try:
            warped = get_perspective_transform(frame, pts)
            input_tensor = transform(warped).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, 1)
                
                # --- STABILITY LOGIC ---
                current_idx = idx.item()
                prediction_history.append(current_idx)
                
                # Only update the stable label if the buffer is full of the same result
                if len(prediction_history) == BUFFER_SIZE:
                    # Check if the most frequent prediction in history is dominant
                    most_common = max(set(prediction_history), key=list(prediction_history).count)
                    if list(prediction_history).count(most_common) > (BUFFER_SIZE * 0.8):
                        stable_label = class_names[most_common]
                        stable_conf = conf.item() * 100

            # Display the stable result
            display_text = f"{stable_label}"
            cv.putText(frame, display_text, (pts[0][0], pts[0][1]-20), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception:
            pass
    else:
        prediction_history.clear()
        stable_label = "Scanning..."

    cv.imshow('Brain Tumor Classifier', frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()