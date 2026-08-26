import cv2
import os
import numpy as np

input_folder = "Task-2/input/ugv_r3_task2"
output_folder = "Task-2/output"

os.makedirs(output_folder, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def average_lane_lines(lines, width, height):

    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:

        x1, y1, x2, y2 = line

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    def make_line(line_data):

        if not line_data:
            return None

        slope, intercept = np.mean(line_data, axis=0)

        y1 = height
        y2 = int(height * 0.55)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, y1, x2, y2)

    left_line = make_line(left_lines)
    right_line = make_line(right_lines)

    return left_line, right_line


for filename in os.listdir(input_folder):

    if not filename.lower().endswith(image_extensions):
        continue

    input_path = os.path.join(input_folder, filename)

    image = cv2.imread(input_path)

    if image is None:
        print("Could not read:", filename)
        continue

    height, width = image.shape[:2]

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.60), int(height * 0.55)),
        (int(width * 0.40), int(height * 0.55))
    ]])

    cv2.fillPoly(mask, polygon, 255)

    roi_edges = cv2.bitwise_and(edges, mask)

    # Detect line segments
    lines = cv2.HoughLinesP(
        roi_edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    # Find left and right lane boundaries
    left_line, right_line = average_lane_lines(
        lines,
        width,
        height
    )

    result = image.copy()

    # Create overlay for drivable area
    overlay = image.copy()

    if left_line is not None and right_line is not None:

        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line

        # Polygon between the two lane boundaries
        drive_area = np.array([[
            (lx1, ly1),
            (lx2, ly2),
            (rx2, ry2),
            (rx1, ry1)
        ]], dtype=np.int32)

        # Fill the drivable area
        cv2.fillPoly(
            overlay,
            drive_area,
            (0, 255, 0)
        )

        # Blend overlay with original image
        result = cv2.addWeighted(
            overlay,
            0.30,
            result,
            0.70,
            0
        )

        # Draw left lane boundary
        cv2.line(
            result,
            (lx1, ly1),
            (lx2, ly2),
            (0, 255, 0),
            5
        )

        # Draw right lane boundary
        cv2.line(
            result,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 0),
            5
        )

    else:

        print("Could not detect both lanes:", filename)

    # Save final output
    output_path = os.path.join(
        output_folder,
        "final_" + filename
    )

    cv2.imwrite(output_path, result)

    print("Final output saved:", output_path)

