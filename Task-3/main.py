
import cv2
import numpy as np
import os


# =========================================================
# CONFIGURATION
# =========================================================

BASE_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_FOLDER = os.path.join(
    BASE_FOLDER,
    "Input"
)

OUTPUT_FOLDER = os.path.join(
    BASE_FOLDER,
    "Output"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def calculate_iou(box1, box2):

    """Calculate Intersection over Union between two boxes."""

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    intersection_x1 = max(x1, a1)
    intersection_y1 = max(y1, b1)
    intersection_x2 = min(x2, a2)
    intersection_y2 = min(y2, b2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area2 = (
        max(0, a2 - a1)
        * max(0, b2 - b1)
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area == 0:
        return 0

    return intersection_area / union_area


def remove_duplicate_boxes(detections):

    """Remove overlapping duplicate detections."""

    final_detections = []

    for detection in detections:

        box = detection["box"]

        duplicate = False

        for existing in final_detections:

            if calculate_iou(
                box,
                existing["box"]
            ) > 0.45:

                duplicate = True
                break

        if not duplicate:

            final_detections.append(
                detection
            )

    return final_detections


# =========================================================
# POTHOLE DETECTION
# =========================================================

def detect_potholes(image):

    """
    Detect bright/white circular pothole-like regions.
    """

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Smooth small noise.
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Adaptive threshold handles illumination changes.
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5
    )

    # Remove small noise.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    image_area = width * height

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.00015:
            continue

        if area > image_area * 0.15:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 8 or h < 8:
            continue

        aspect_ratio = w / float(h)

        if (
            aspect_ratio < 0.30
            or aspect_ratio > 3.0
        ):
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )

        if circularity < 0.25:
            continue

        detections.append({
            "box": (
                x,
                y,
                x + w,
                y + h
            ),
            "type": "Pothole",
            "confidence": circularity
        })

    return detections


# =========================================================
# OBSTACLE DETECTION
# =========================================================

def detect_obstacles(image):

    """
    Detect colored obstacles on the asphalt road.

    The Task-3 images contain obstacles with strong colors
    such as yellow, blue and green.

    Asphalt and lane markings have very low saturation,
    therefore HSV saturation is used to separate obstacles
    from the road.
    """

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # COLORED OBJECT MASK
    # -----------------------------------------------------

    # Saturated pixels correspond to colored obstacles.
    #
    # Gray asphalt:
    #     Saturation is close to 0.
    #
    # White lane markings:
    #     Saturation is also close to 0.
    #
    # Yellow / green / blue obstacles:
    #     Saturation is high.

    colored_mask = cv2.inRange(
        hsv,
        np.array([0, 70, 30]),
        np.array([179, 255, 255])
    )

    # -----------------------------------------------------
    # REMOVE SMALL NOISE
    # -----------------------------------------------------

    small_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    colored_mask = cv2.morphologyEx(
        colored_mask,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # CONNECT SMALL GAPS
    # -----------------------------------------------------

    close_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    colored_mask = cv2.morphologyEx(
        colored_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # FIND COLORED OBJECT CONTOURS
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        colored_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    image_area = width * height

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore tiny colored noise.
        if area < image_area * 0.0008:
            continue

        # Reject an accidentally detected huge region.
        if area > image_area * 0.20:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Minimum object size.
        if w < width * 0.015:
            continue

        if h < height * 0.015:
            continue

        # -------------------------------------------------
        # ASPECT RATIO
        # -------------------------------------------------

        aspect_ratio = w / float(h)

        # Reject extremely thin objects.
        if aspect_ratio > 7.0:
            continue

        if aspect_ratio < 0.10:
            continue

        # -------------------------------------------------
        # EXTENT
        # -------------------------------------------------

        rectangle_area = w * h

        if rectangle_area == 0:
            continue

        extent = (
            area
            / float(rectangle_area)
        )

        # Colored obstacle shapes can be irregular,
        # therefore use a low threshold.
        if extent < 0.03:
            continue

        # -------------------------------------------------
        # SOLIDITY
        # -------------------------------------------------

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area > 0:

            solidity = (
                area
                / hull_area
            )

        else:

            solidity = 0

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = (
            0.40 * min(extent, 1.0)
            +
            0.30 * min(solidity, 1.0)
            +
            0.30 * min(
                area
                / (image_area * 0.08),
                1.0
            )
        )

        # -------------------------------------------------
        # SAVE DETECTION
        # -------------------------------------------------

        detections.append({
            "box": (
                x,
                y,
                x + w,
                y + h
            ),
            "type": "Obstacle",
            "confidence": confidence
        })

    return detections


# =========================================================
# COMBINE DETECTIONS
# =========================================================

def detect_objects(image):

    potholes = detect_potholes(
        image
    )

    obstacles = detect_obstacles(
        image
    )

    # Potholes should not also be counted
    # as obstacles.

    pothole_boxes = [
        detection["box"]
        for detection in potholes
    ]

    filtered_obstacles = []

    for obstacle in obstacles:

        overlap = False

        for pothole_box in pothole_boxes:

            if calculate_iou(
                obstacle["box"],
                pothole_box
            ) > 0.20:

                overlap = True
                break

        if not overlap:

            filtered_obstacles.append(
                obstacle
            )

    detections = (
        potholes
        + filtered_obstacles
    )

    detections = remove_duplicate_boxes(
        detections
    )

    return detections


# =========================================================
# DRAW RESULTS
# =========================================================

def draw_results(image, detections):

    result = image.copy()

    height, width = image.shape[:2]

    pothole_count = 0
    obstacle_count = 0

    thickness = max(
        2,
        width // 500
    )

    font_scale = max(
        0.5,
        width / 1600
    )

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        object_type = detection["type"]

        if object_type == "Pothole":

            pothole_count += 1

        else:

            obstacle_count += 1

        # Keep coordinates inside image.

        x1 = max(
            0,
            min(width - 1, x1)
        )

        y1 = max(
            0,
            min(height - 1, y1)
        )

        x2 = max(
            0,
            min(width - 1, x2)
        )

        y2 = max(
            0,
            min(height - 1, y2)
        )

        # -------------------------------------------------
        # COLORS
        # -------------------------------------------------

        # Pothole = RED
        # Obstacle = BLUE

        if object_type == "Pothole":

            color = (
                0,
                0,
                255
            )

        else:

            color = (
                255,
                0,
                0
            )

        # -------------------------------------------------
        # DRAW BOX
        # -------------------------------------------------

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # -------------------------------------------------
        # DRAW LABEL + COORDINATES
        # -------------------------------------------------

        coordinate_text = (
            f"{object_type}: "
            f"({x1},{y1})-({x2},{y2})"
        )

        text_y = max(
            20,
            y1 - 8
        )

        cv2.putText(
            result,
            coordinate_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(
                1,
                thickness // 2
            ),
            cv2.LINE_AA
        )

    # =====================================================
    # SUMMARY PANEL
    # =====================================================

    summary = (
        f"Potholes: {pothole_count}    "
        f"Obstacles: {obstacle_count}"
    )

    panel_width = min(
        width - 10,
        int(width * 0.60)
    )

    panel_height = int(
        height * 0.075
    )

    cv2.rectangle(
        result,
        (10, 10),
        (
            panel_width,
            panel_height
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result,
        summary,
        (
            20,
            int(height * 0.055)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(
            1,
            thickness // 2
        ),
        cv2.LINE_AA
    )

    return (
        result,
        pothole_count,
        obstacle_count
    )


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(
    input_path,
    output_path
):

    image = cv2.imread(
        input_path
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return False

    detections = detect_objects(
        image
    )

    (
        result,
        pothole_count,
        obstacle_count
    ) = draw_results(
        image,
        detections
    )

    saved = cv2.imwrite(
        output_path,
        result
    )

    if not saved:

        print(
            "ERROR: Could not save output."
        )

        return False

    print(
        f"  Potholes detected : "
        f"{pothole_count}"
    )

    print(
        f"  Obstacles detected: "
        f"{obstacle_count}"
    )

    print(
        f"  Saved: {output_path}"
    )

    return True


# =========================================================
# MAIN PROGRAM
# =========================================================

print()

print(
    "======================================"
)

print(
    "Task-3 Obstacle & Pothole Detection"
)

print(
    "======================================"
)

print(
    f"Input folder : {INPUT_FOLDER}"
)

print(
    f"Output folder: {OUTPUT_FOLDER}"
)

print(
    "======================================"
)


# ---------------------------------------------------------
# CHECK INPUT FOLDER
# ---------------------------------------------------------

if not os.path.exists(
    INPUT_FOLDER
):

    print()

    print(
        "ERROR: Input folder does not exist."
    )

    print(
        "Create Task-3/Input and place "
        "the images there."
    )

    raise SystemExit


# ---------------------------------------------------------
# SUPPORTED IMAGE FORMATS
# ---------------------------------------------------------

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp"
)


# ---------------------------------------------------------
# GET ALL INPUT IMAGES
# ---------------------------------------------------------

files = sorted(
    [
        filename
        for filename in os.listdir(
            INPUT_FOLDER
        )
        if filename.lower().endswith(
            valid_extensions
        )
    ]
)


print(
    f"Images found: {len(files)}"
)

print(
    "======================================"
)


if len(files) == 0:

    print()

    print(
        "No images found."
    )

    print(
        "Put the Task-3 images inside:"
    )

    print(
        INPUT_FOLDER
    )

    raise SystemExit


# ---------------------------------------------------------
# PROCESS ALL IMAGES
# ---------------------------------------------------------

processed_count = 0

for filename in files:

    print()

    print(
        f"Processing: {filename}"
    )

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    name, extension = os.path.splitext(
        filename
    )

    output_filename = (
        name
        + "_detected"
        + extension
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )

    try:

        success = process_image(
            input_path,
            output_path
        )

        if success:

            processed_count += 1

    except Exception as error:

        print(
            f"ERROR while processing: "
            f"{filename}"
        )

        print(
            f"Reason: {error}"
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()

print(
    "======================================"
)

print(
    "Obstacle & Pothole Detection Completed"
)

print(
    f"Total Images Processed: "
    f"{processed_count}"
)

print(
    "======================================"
)



