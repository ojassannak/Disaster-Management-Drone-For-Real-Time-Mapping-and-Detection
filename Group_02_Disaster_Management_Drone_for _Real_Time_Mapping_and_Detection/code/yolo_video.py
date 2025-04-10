import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the video file
video_path = "istockphoto-1425778227-640_adpp_is.mp4"  # Change this to your video filename
cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer to save output
output_video = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extract detection information
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on detected objects
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the processed frame to output video
    output_video.write(frame)

    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

#####________________________________________________________________________#####



# import cv2
# import numpy as np

# # Load YOLO model with CUDA (GPU acceleration)
# net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use CPU instead of CUDA
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Run on CPU

# # Get output layers
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Load class labels
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Open webcam instead of video file
# # cap = cv2.VideoCapture(1)  # 0 = default webcam

# video_path = "istockphoto-1425778227-640_adpp_is.mp4"  # Change this to your video filename
# cap = cv2.VideoCapture(video_path)

# # Get video properties
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Define video writer to save output
# output_video = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     height, width, _ = frame.shape

#     # Preprocess the frame for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Extract detection information
#     class_ids, confidences, boxes = [], [], []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.3:  # Lower confidence threshold for more detections
#                 center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
#                 x, y = center_x - w // 2, center_y - h // 2
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply Non-Maximum Suppression (NMS) with improved threshold
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)  # Lower NMS threshold

#     # Draw bounding boxes
#     colors = np.random.uniform(0, 255, size=(len(classes), 3))
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
#             color = colors[class_ids[i]]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Save processed frame to output video
#     output_video.write(frame)

#     # Display the frame
#     cv2.imshow("YOLO Object Detection", frame)

#     # Press 'q' to exit early
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# output_video.release()
# cv2.destroyAllWindows()













