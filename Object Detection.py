import cv2
import numpy as np

# Step 1: Load YOLO
cfg_file = "yolov4.cfg"
weights_file = "yolov4.weights"
names_file = "coco.names"

INPUT_SIZE = 608
CONF_THRESH = 0.25
NMS_THRESH = 0.50

net = cv2.dnn.readNet(weights_file, cfg_file)

layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()
unconnected = unconnected.flatten() if hasattr(unconnected, "flatten") else unconnected
output_layers = [layer_names[i - 1] for i in unconnected]

# Load class labels
with open(names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Letterbox function (same as yours)
def letterbox_image(img, new_size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y

# Step 2: Start Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Preprocess
    letterboxed, scale, pad_x, pad_y = letterbox_image(frame, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), (0, 0, 0), swapRB=True, crop=False)

    # Forward pass
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            objectness = float(detection[4])
            confidence = objectness * class_score

            if confidence > CONF_THRESH:
                center_x = detection[0] * INPUT_SIZE
                center_y = detection[1] * INPUT_SIZE
                w = detection[2] * INPUT_SIZE
                h = detection[3] * INPUT_SIZE

                x = (center_x - w / 2 - pad_x) / scale
                y = (center_y - h / 2 - pad_y) / scale
                w = w / scale
                h = h / scale

                x = int(max(0, min(orig_w - 1, x)))
                y = int(max(0, min(orig_h - 1, y)))
                w = int(max(1, min(orig_w - x, w)))
                h = int(max(1, min(orig_h - y, h)))

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    indices = indices.flatten() if len(indices) > 0 else []

    # Draw boxes
    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = f"{label} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        ty = y - 10
        if ty - th - baseline < 0:
            ty = y + th + 10

        tx = x
        if tx + tw >= orig_w:
            tx = max(0, orig_w - tw - 1)

        cv2.rectangle(frame, (tx, ty - th - baseline), (tx + tw, ty + baseline), (0, 255, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show output
    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()