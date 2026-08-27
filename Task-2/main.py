import cv2
import numpy as np
import os
import warnings


# =========================================================
# FUNCTION 1: CREATE FINAL LANE COORDINATES
# =========================================================
def make_coordinates(image, line_parameters):

    slope, intercept = line_parameters

    height = image.shape[0]
    width = image.shape[1]

    y1 = int(height * 0.90)
    y2 = int(height * 0.55)

    if abs(slope) < 0.001:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))

    return np.array([x1, y1, x2, y2])


# =========================================================
# FUNCTION 2: SELECT LEFT / RIGHT LANE CANDIDATE
# =========================================================
def select_lane_candidate(candidates, side, width):

    if len(candidates) == 0:
        return None

    if side == "left":
        reference_x = max(
            candidate[3]
            for candidate in candidates
        )

    else:
        reference_x = min(
            candidate[3]
            for candidate in candidates
        )

    close_candidates = [
        candidate
        for candidate in candidates
        if abs(candidate[3] - reference_x) < width * 0.10
    ]

    if len(close_candidates) == 0:
        return None

    values = np.array(close_candidates)

    return np.average(
        values[:, :2],
        axis=0,
        weights=values[:, 2]
    )


# =========================================================
# FUNCTION 3: FIND LEFT AND RIGHT LANES
# =========================================================
def average_lane_lines(image, lines):

    left_lines = []
    right_lines = []

    height, width = image.shape[:2]
    reference_y = height * 0.80

    if lines is None:
        return None, None

    # -----------------------------------------------------
    # IMPORTANT FIX:
    # Normalize OpenCV Hough output to N x 4
    # -----------------------------------------------------
    lines = np.asarray(lines).reshape(-1, 4)

    for x1, y1, x2, y2 in lines:

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        if abs(x2 - x1) < 1e-6:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if not 0.45 < abs(slope) < 5.0:
            continue

        intercept = y1 - slope * x1

        middle_x = (x1 + x2) / 2

        x_at_reference = (
            reference_y - intercept
        ) / slope

        length = np.hypot(
            x2 - x1,
            y2 - y1
        )

        candidate = (
            slope,
            intercept,
            length,
            x_at_reference
        )

        # -------------------------------------------------
        # LEFT LANE
        # -------------------------------------------------
        if (
            slope < 0
            and middle_x < width * 0.60
            and width * 0.05 < x_at_reference < width * 0.60
        ):
            left_lines.append(candidate)

        # -------------------------------------------------
        # RIGHT LANE
        # -------------------------------------------------
        elif (
            slope > 0
            and middle_x > width * 0.40
            and width * 0.40 < x_at_reference < width * 0.95
        ):
            right_lines.append(candidate)

    left_average = select_lane_candidate(
        left_lines,
        "left",
        width
    )

    right_average = select_lane_candidate(
        right_lines,
        "right",
        width
    )

    left_lane = None
    right_lane = None

    if left_average is not None:

        left_lane = make_coordinates(
            image,
            left_average
        )

    if right_average is not None:

        right_lane = make_coordinates(
            image,
            right_average
        )

    return left_lane, right_lane


# =========================================================
# FUNCTION 4: QUADRATIC RANSAC
# =========================================================
def fit_quadratic_ransac(points, random_generator):

    if len(points) < 12:
        return None, None

    best_model = None

    for _ in range(450):

        sample = points[
            random_generator.choice(
                len(points),
                3,
                replace=False
            )
        ]

        if (
            len(np.unique(sample[:, 0])) < 3
            or np.ptp(sample[:, 0]) < 0.12
        ):
            continue

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            coefficients = np.polyfit(
                sample[:, 0],
                sample[:, 1],
                2
            )

        predicted_x = np.polyval(
            coefficients,
            points[:, 0]
        )

        inliers = (
            np.abs(
                predicted_x - points[:, 1]
            ) < 0.025
        )

        if (
            np.count_nonzero(inliers) < 10
            or np.ptp(points[inliers, 0]) < 0.17
        ):
            continue

        score = (
            np.count_nonzero(inliers)
            + 2 * np.ptp(points[inliers, 0])
        )

        if best_model is None or score > best_model[0]:

            best_model = (
                score,
                coefficients,
                inliers
            )

    if best_model is None:
        return None, None

    coefficients = best_model[1]
    inliers = best_model[2]

    for _ in range(3):

        if np.count_nonzero(inliers) < 3:
            return None, None

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            coefficients = np.polyfit(
                points[inliers, 0],
                points[inliers, 1],
                2
            )

        inliers = (
            np.abs(
                np.polyval(
                    coefficients,
                    points[:, 0]
                )
                - points[:, 1]
            ) < 0.028
        )

    return coefficients, inliers


# =========================================================
# FUNCTION 5: CURVED LANE DETECTION
# =========================================================
def find_curved_lanes(image, edges):

    height, width = image.shape[:2]

    curve_mask = np.zeros_like(edges)

    curve_polygon = np.array([[
        (0, int(height * 0.90)),
        (int(width * 0.12), int(height * 0.35)),
        (int(width * 0.85), int(height * 0.35)),
        (width - 1, int(height * 0.90))
    ]], dtype=np.int32)

    cv2.fillPoly(
        curve_mask,
        curve_polygon,
        255
    )

    curve_edges = cv2.bitwise_and(
        edges,
        curve_mask
    )

    curve_lines = cv2.HoughLinesP(
        curve_edges,
        rho=1,
        theta=np.pi / 360,
        threshold=15,
        minLineLength=15,
        maxLineGap=15
    )

    if curve_lines is None:
        return None

    # -----------------------------------------------------
    # IMPORTANT FIX
    # -----------------------------------------------------
    curve_lines = np.asarray(
        curve_lines
    ).reshape(-1, 4)

    points = []

    for x1, y1, x2, y2 in curve_lines:

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        if abs(x2 - x1) < 1e-6:
            continue

        slope = (y2 - y1) / (x2 - x1)

        length = np.hypot(
            x2 - x1,
            y2 - y1
        )

        if (
            -5.0 < slope < -0.18
            and length > 20
        ):

            points.extend([
                (y1 / height, x1 / width),
                (y2 / height, x2 / width)
            ])

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < 24:
        return None

    random_generator = np.random.default_rng(7)

    first_curve, first_inliers = fit_quadratic_ransac(
        points,
        random_generator
    )

    if first_curve is None:
        return None

    second_curve, _ = fit_quadratic_ransac(
        points[~first_inliers],
        random_generator
    )

    if second_curve is None:
        return None

    y_values = np.arange(
        int(height * 0.45),
        int(height * 0.90) + 1
    )

    first_path = np.stack([
        np.clip(
            np.polyval(
                first_curve,
                y_values / height
            ) * width,
            0,
            width - 1
        ).astype(np.int32),
        y_values
    ], axis=1)

    second_path = np.stack([
        np.clip(
            np.polyval(
                second_curve,
                y_values / height
            ) * width,
            0,
            width - 1
        ).astype(np.int32),
        y_values
    ], axis=1)

    if np.mean(first_path[-20:, 0]) > np.mean(
        second_path[-20:, 0]
    ):

        first_path, second_path = (
            second_path,
            first_path
        )

    if (
        np.any(
            first_path[:, 0]
            >= second_path[:, 0]
        )
        or
        second_path[-1, 0]
        - first_path[-1, 0]
        < width * 0.15
    ):
        return None

    return first_path, second_path


# =========================================================
# FUNCTION 6: CREATE LINE AT SPECIFIC Y VALUES
# =========================================================
def make_line_at_y(
    image,
    slope,
    intercept,
    y_bottom,
    y_top
):

    height, width = image.shape[:2]

    if abs(slope) < 0.001:
        return None

    bottom_x = int(
        np.clip(
            (y_bottom - intercept) / slope,
            0,
            width - 1
        )
    )

    top_x = int(
        np.clip(
            (y_top - intercept) / slope,
            0,
            width - 1
        )
    )

    return np.array([
        bottom_x,
        int(y_bottom),
        top_x,
        int(y_top)
    ])


# =========================================================
# FUNCTION 7: SHORTEN CROSSING LANE PAIR
# =========================================================
def shorten_crossing_lane_pair(
    image,
    left_lane,
    right_lane
):

    height, width = image.shape[:2]

    lx1, ly1, lx2, ly2 = left_lane
    rx1, ry1, rx2, ry2 = right_lane

    if ly2 == ly1 or ry2 == ry1:
        return None

    left_a = (
        (lx2 - lx1)
        / (ly2 - ly1)
    )

    left_b = (
        lx1
        - left_a * ly1
    )

    right_a = (
        (rx2 - rx1)
        / (ry2 - ry1)
    )

    right_b = (
        rx1
        - right_a * ry1
    )

    if abs(left_a - right_a) < 0.001:
        return None

    intersection_y = (
        right_b - left_b
    ) / (
        left_a - right_a
    )

    y_bottom = min(
        ly1,
        ry1
    )

    y_top = int(
        max(
            height * 0.62,
            intersection_y + height * 0.04
        )
    )

    if y_top >= y_bottom - height * 0.08:
        return None

    if abs(left_a) < 0.001 or abs(right_a) < 0.001:
        return None

    safe_left = make_line_at_y(
        image,
        1 / left_a,
        -left_b / left_a,
        y_bottom,
        y_top
    )

    safe_right = make_line_at_y(
        image,
        1 / right_a,
        -right_b / right_a,
        y_bottom,
        y_top
    )

    if safe_left is None or safe_right is None:
        return None

    if (
        safe_left[0] + width * 0.08
        >= safe_right[0]
        or safe_left[2]
        >= safe_right[2]
    ):
        return None

    return safe_left, safe_right


# =========================================================
# FUNCTION 8: DASHBOARD CURVE FALLBACK
# =========================================================
def find_dashboard_curve_lane_pair(image):

    height, width = image.shape[:2]

    hls = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HLS
    )

    white_mask = cv2.inRange(
        hls,
        np.array([0, 90, 0]),
        np.array([180, 255, 130])
    )

    road_mask = np.zeros_like(
        white_mask
    )

    cv2.fillPoly(
        road_mask,
        np.array([[
            (int(width * 0.01), int(height * 0.72)),
            (int(width * 0.15), int(height * 0.30)),
            (int(width * 0.85), int(height * 0.30)),
            (int(width * 0.99), int(height * 0.72))
        ]], dtype=np.int32),
        255
    )

    lines = cv2.HoughLinesP(
        cv2.bitwise_and(
            white_mask,
            road_mask
        ),
        rho=1,
        theta=np.pi / 360,
        threshold=8,
        minLineLength=8,
        maxLineGap=10
    )

    if lines is None:
        return None

    # -----------------------------------------------------
    # IMPORTANT FIX
    # -----------------------------------------------------
    lines = np.asarray(lines).reshape(-1, 4)

    left_points = []
    right_candidates = []

    for x1, y1, x2, y2 in lines:

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        if abs(x2 - x1) < 1e-6:
            continue

        slope = (
            (y2 - y1)
            / (x2 - x1)
        )

        intercept = (
            y1
            - slope * x1
        )

        length = np.hypot(
            x2 - x1,
            y2 - y1
        )

        middle_x = (
            x1 + x2
        ) / 2

        middle_y = (
            y1 + y2
        ) / 2

        if (
            abs(slope) > 1.2
            and width * 0.22 < middle_x < width * 0.42
            and height * 0.33 < middle_y < height * 0.62
            and length > 12
        ):

            left_points.extend([
                (x1, y1),
                (x2, y2)
            ])

        if (
            0.25 < slope < 0.75
            and length > width * 0.30
            and height * 0.28 < middle_y < height * 0.60
        ):

            right_candidates.append(
                (
                    slope,
                    intercept,
                    length
                )
            )

    if (
        len(left_points) < 8
        or len(right_candidates) == 0
    ):
        return None

    left_points = np.asarray(
        left_points,
        dtype=float
    )

    left_a, left_b = np.polyfit(
        left_points[:, 1],
        left_points[:, 0],
        1
    )

    right_slope, right_intercept, _ = max(
        right_candidates,
        key=lambda candidate: candidate[2]
    )

    y_bottom = int(
        height * 0.68
    )

    y_top = int(
        height * 0.32
    )

    if abs(left_a) < 0.001:
        return None

    left_lane = make_line_at_y(
        image,
        1 / left_a,
        -left_b / left_a,
        y_bottom,
        y_top
    )

    right_lane = make_line_at_y(
        image,
        right_slope,
        right_intercept,
        y_bottom,
        y_top
    )

    if (
        left_lane is None
        or right_lane is None
    ):
        return None

    if (
        left_lane[0] + width * 0.12
        >= right_lane[0]
        or left_lane[2]
        >= right_lane[2]
    ):
        return None

    return left_lane, right_lane


# =========================================================
# FUNCTION 9: OCCLUDED LANE FALLBACK
# =========================================================
def find_occluded_lane_pair(image):

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.createCLAHE(
        clipLimit=3,
        tileGridSize=(8, 8)
    ).apply(gray)

    edges = cv2.Canny(
        cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0
        ),
        25,
        90
    )

    road_mask = np.zeros_like(
        edges
    )

    cv2.fillPoly(
        road_mask,
        np.array([[
            (int(width * 0.01), int(height * 0.83)),
            (int(width * 0.10), int(height * 0.38)),
            (int(width * 0.90), int(height * 0.38)),
            (int(width * 0.99), int(height * 0.83))
        ]], dtype=np.int32),
        255
    )

    lines = cv2.HoughLinesP(
        cv2.bitwise_and(
            edges,
            road_mask
        ),
        rho=1,
        theta=np.pi / 360,
        threshold=8,
        minLineLength=8,
        maxLineGap=8
    )

    if lines is None:
        return None

    # -----------------------------------------------------
    # IMPORTANT FIX
    # -----------------------------------------------------
    lines = np.asarray(lines).reshape(-1, 4)

    left_candidates = []
    right_candidates = []

    for x1, y1, x2, y2 in lines:

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        if abs(x2 - x1) < 1e-6:
            continue

        slope = (
            (y2 - y1)
            / (x2 - x1)
        )

        intercept = (
            y1
            - slope * x1
        )

        length = np.hypot(
            x2 - x1,
            y2 - y1
        )

        if (
            -1.0 < slope < -0.25
            and max(x1, x2) < width * 0.45
            and min(y1, y2) > height * 0.50
            and length > 12
        ):

            left_candidates.append(
                (
                    slope,
                    intercept,
                    length
                )
            )

        elif (
            0.50 < slope < 2.0
            and min(x1, x2) > width * 0.60
            and min(y1, y2) > height * 0.60
            and length > 12
        ):

            right_candidates.append(
                (
                    slope,
                    intercept,
                    length
                )
            )

    if (
        len(left_candidates) == 0
        or len(right_candidates) == 0
    ):
        return None

    left_slope, left_intercept, _ = max(
        left_candidates,
        key=lambda candidate: candidate[2]
    )

    right_slope, right_intercept, _ = max(
        right_candidates,
        key=lambda candidate: candidate[2]
    )

    y_bottom = int(
        height * 0.80
    )

    y_top = int(
        height * 0.55
    )

    left_lane = make_line_at_y(
        image,
        left_slope,
        left_intercept,
        y_bottom,
        y_top
    )

    right_lane = make_line_at_y(
        image,
        right_slope,
        right_intercept,
        y_bottom,
        y_top
    )

    if (
        left_lane is None
        or right_lane is None
    ):
        return None

    if (
        left_lane[0] + width * 0.10
        >= right_lane[0]
        or left_lane[2]
        >= right_lane[2]
    ):
        return None

    return left_lane, right_lane


# =========================================================
# FUNCTION 10: PORTRAIT FALLBACK
# =========================================================
def find_portrait_lane_pair(image):

    height, width = image.shape[:2]

    aspect_ratio = height / width

    if not 1.05 < aspect_ratio:
        return None

    if aspect_ratio < 1.40:

        return find_dashboard_curve_lane_pair(
            image
        )

    return find_occluded_lane_pair(
        image
    )


# =========================================================
# FUNCTION 11: PROCESS ONE IMAGE
# =========================================================
def process_image(image):

    # -----------------------------------------------------
    # STEP 1: GRAYSCALE
    # -----------------------------------------------------
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # STEP 2: GAUSSIAN BLUR
    # -----------------------------------------------------
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # STEP 3: CANNY EDGE DETECTION
    # -----------------------------------------------------
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # -----------------------------------------------------
    # STEP 4: REGION OF INTEREST
    # -----------------------------------------------------
    height = edges.shape[0]
    width = edges.shape[1]

    mask = np.zeros_like(
        edges
    )

    polygon = np.array([[
        (int(width * 0.03), int(height * 0.90)),
        (int(width * 0.25), int(height * 0.47)),
        (int(width * 0.75), int(height * 0.47)),
        (int(width * 0.97), int(height * 0.90))
    ]], dtype=np.int32)

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    roi_edges = cv2.bitwise_and(
        edges,
        mask
    )

    # -----------------------------------------------------
    # STEP 5: HOUGH LINE TRANSFORM
    # -----------------------------------------------------
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=45
    )

    # -----------------------------------------------------
    # STEP 6: FIND LEFT + RIGHT LANES
    # -----------------------------------------------------
    left_lane, right_lane = average_lane_lines(
        image,
        lines
    )

    straight_lanes = None

    if (
        left_lane is not None
        and right_lane is not None
    ):

        lx1, ly1, lx2, ly2 = left_lane
        rx1, ry1, rx2, ry2 = right_lane

        if (
            lx1 + width * 0.08 < rx1
            and lx2 < rx2
        ):

            straight_lanes = (
                left_lane,
                right_lane
            )

        else:

            straight_lanes = shorten_crossing_lane_pair(
                image,
                left_lane,
                right_lane
            )

    # -----------------------------------------------------
    # STEP 7: CURVED ROAD FALLBACK
    # -----------------------------------------------------
    curved_lanes = None

    if (
        straight_lanes is None
        and (
            left_lane is None
            or right_lane is None
        )
    ):

        curved_lanes = find_curved_lanes(
            image,
            edges
        )

    # -----------------------------------------------------
    # STEP 8: OCCLUSION / PORTRAIT FALLBACK
    # -----------------------------------------------------
    if (
        straight_lanes is None
        and curved_lanes is None
    ):

        straight_lanes = find_portrait_lane_pair(
            image
        )

    # -----------------------------------------------------
    # STEP 9: PREPARE OUTPUT
    # -----------------------------------------------------
    lane_image = image.copy()

    lane_detected = False

    line_thickness = max(
        3,
        width // 160
    )

    text_origin = (
        max(15, width // 40),
        max(30, height // 14)
    )

    text_scale = max(
        0.45,
        width / 1200
    )

    # -----------------------------------------------------
    # CURVED LANES
    # -----------------------------------------------------
    if curved_lanes is not None:

        left_curve, right_curve = curved_lanes

        overlay = image.copy()

        drivable_area = np.array([
            np.vstack([
                left_curve,
                right_curve[::-1]
            ])
        ], dtype=np.int32)

        cv2.fillPoly(
            overlay,
            drivable_area,
            (0, 255, 0)
        )

        lane_image = cv2.addWeighted(
            overlay,
            0.30,
            image,
            0.70,
            0
        )

        cv2.polylines(
            lane_image,
            [left_curve],
            False,
            (0, 0, 255),
            line_thickness
        )

        cv2.polylines(
            lane_image,
            [right_curve],
            False,
            (0, 0, 255),
            line_thickness
        )

        lane_detected = True

    # -----------------------------------------------------
    # STRAIGHT LANES
    # -----------------------------------------------------
    elif straight_lanes is not None:

        left_lane, right_lane = straight_lanes

        lx1, ly1, lx2, ly2 = left_lane
        rx1, ry1, rx2, ry2 = right_lane

        if (
            lx1 + width * 0.08 < rx1
            and lx2 < rx2
        ):

            overlay = image.copy()

            drivable_area = np.array([[
                (lx1, ly1),
                (lx2, ly2),
                (rx2, ry2),
                (rx1, ry1)
            ]], dtype=np.int32)

            cv2.fillPoly(
                overlay,
                drivable_area,
                (0, 255, 0)
            )

            lane_image = cv2.addWeighted(
                overlay,
                0.30,
                image,
                0.70,
                0
            )

            cv2.line(
                lane_image,
                (lx1, ly1),
                (lx2, ly2),
                (0, 0, 255),
                line_thickness
            )

            cv2.line(
                lane_image,
                (rx1, ry1),
                (rx2, ry2),
                (0, 0, 255),
                line_thickness
            )

            lane_detected = True

    # -----------------------------------------------------
    # INCOMPLETE DETECTION
    # -----------------------------------------------------
    if not lane_detected:

        for lane in (
            left_lane,
            right_lane
        ):

            if lane is not None:

                x1, y1, x2, y2 = lane

                cv2.line(
                    lane_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    line_thickness
                )

        cv2.putText(
            lane_image,
            "Lane Detection Incomplete",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 255),
            max(
                1,
                line_thickness // 2
            ),
            cv2.LINE_AA
        )

    # -----------------------------------------------------
    # SUCCESSFUL DETECTION
    # -----------------------------------------------------
    else:

        cv2.putText(
            lane_image,
            "Drivable Area",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 0),
            max(
                1,
                line_thickness // 2
            ),
            cv2.LINE_AA
        )

    return lane_image


# =========================================================
# MAIN PROGRAM
# =========================================================

base_folder = os.path.dirname(
    os.path.abspath(__file__)
)


print("\n======================================")
print("Task-2 Lane Detection")
print("======================================")


# ---------------------------------------------------------
# INPUT / OUTPUT
# ---------------------------------------------------------
input_folder = os.path.join(
    base_folder,
    "Input"
)

output_folder = os.path.join(
    base_folder,
    "Output"
)


os.makedirs(
    output_folder,
    exist_ok=True
)


# ---------------------------------------------------------
# CHECK INPUT
# ---------------------------------------------------------
if not os.path.exists(input_folder):

    print("ERROR: Input folder does not exist:")
    print(input_folder)

    input("\nPress Enter to exit...")
    raise SystemExit


# ---------------------------------------------------------
# SEARCH FOR IMAGES RECURSIVELY
# ---------------------------------------------------------
valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_paths = []


for root, dirs, files in os.walk(
    input_folder
):

    for filename in files:

        if filename.lower().endswith(
            valid_extensions
        ):

            image_paths.append(
                os.path.join(
                    root,
                    filename
                )
            )


image_paths.sort()


# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------
print(
    "Input folder:",
    input_folder
)

print(
    "Output folder:",
    output_folder
)

print(
    "Images found:",
    len(image_paths)
)

print("======================================")


# ---------------------------------------------------------
# NO IMAGES
# ---------------------------------------------------------
if len(image_paths) == 0:

    print("\nERROR: No images were found.")

    print("\nSearched recursively inside:")
    print(input_folder)

    print("\nSupported formats:")
    print(
        ".jpg  .jpeg  .png  .bmp  .webp"
    )

    input("\nPress Enter to exit...")
    raise SystemExit


# ---------------------------------------------------------
# PROCESS ALL IMAGES
# ---------------------------------------------------------
processed_count = 0


for image_path in image_paths:

    filename = os.path.basename(
        image_path
    )

    print(
        "\nProcessing:",
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Could not read:",
            filename
        )

        continue

    try:

        result = process_image(
            image
        )

    except Exception as error:

        print(
            "ERROR while processing:",
            filename
        )

        print(
            "Reason:",
            error
        )

        continue

    # -----------------------------------------------------
    # OUTPUT NAME
    # -----------------------------------------------------
    name, extension = os.path.splitext(
        filename
    )

    output_filename = (
        name
        + "_lane_detected"
        + ".png"
    )

    output_path = os.path.join(
        output_folder,
        output_filename
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    saved = cv2.imwrite(
        output_path,
        result
    )

    if saved:

        print(
            "Saved:",
            output_filename
        )

        processed_count += 1

    else:

        print(
            "ERROR: Could not save:",
            output_filename
        )


# =========================================================
# FINAL RESULT
# =========================================================

print("\n======================================")
print("Lane Detection Completed")
print(
    "Total Images Processed:",
    processed_count
)
print("======================================")

print("\nOutput folder:")
print(output_folder)

input("\nPress Enter to exit...")