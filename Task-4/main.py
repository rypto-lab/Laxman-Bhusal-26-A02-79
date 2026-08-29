import cv2
import numpy as np
import os
import heapq
import math


# ================================================================
# TASK 4 - AERIAL PATH PLANNING
# ================================================================
#
# Idea used in this program:
# 1. Detect the grey road from the aerial image.
# 2. Treat coloured objects and black potholes as unsafe areas.
# 3. Put checkpoints around the inside of the closed track.
# 4. Use A* to join every checkpoint without leaving the safe road.
#
# The images in this task are 1200 x 1200.  A* is run on a smaller
# grid so that the code stays fast and easy to understand.


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_FOLDER, "Input")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "Output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


GRID_SIZE = 4               # One A* cell represents 4 x 4 image pixels.
CHECKPOINT_COUNT = 14       # More checkpoints make the loop smoother.


# ----------------------------------------------------------------
# HELPER FUNCTIONS FOR IMAGE MASKS
# ----------------------------------------------------------------

def largest_component(mask):
    """Keep only the biggest white connected area in a black/white mask."""

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if number <= 1:
        return np.zeros_like(mask)

    # Label 0 is the black background, so start checking from label 1.
    biggest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    clean_mask = np.zeros_like(mask)
    clean_mask[labels == biggest_label] = 255

    return clean_mask


def detect_road(image):
    """Return a mask in which white pixels are the road."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The road has a grey value close to 92.  Blurring first removes the
    # texture/noise present in some of the supplied images.
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    road_mask = cv2.inRange(blurred, 90, 94)

    # Join very tiny gaps caused by image compression.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )
    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        small_kernel
    )

    # Other parts of the picture can have a similar grey shade.  The road
    # is the largest connected grey shape, so keep only that shape.
    road_mask = largest_component(road_mask)

    # Some potholes/objects make narrow holes in the detected road.  Close
    # those holes here to get one clean course shape.  They are removed again
    # later by detect_hazards(), so the final path still avoids them.
    course_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (41, 41)
    )
    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        course_kernel
    )

    return road_mask


def keep_object_blobs(mask):
    """Remove very small noise from an object/hazard mask."""

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)

        # Real objects in these images are much larger than small texture
        # marks on the road.  The upper limit ignores a large background area.
        if area < 25 or area > 8000:
            continue

        if width < 3 or height < 3:
            continue

        cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    return clean_mask


def detect_hazards(image, road_mask):
    """Detect coloured obstacles, light blocks, and dark potholes."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Coloured obstacles have high saturation.  Potholes are dark, while
    # the small square obstacles are light but not pure-white START text.
    coloured = hsv[:, :, 1] > 50
    dark = gray < 72
    light_object = (gray > 125) & (gray < 245)

    hazard_mask = np.zeros_like(gray)
    hazard_mask[coloured | dark | light_object] = 255

    # We only care about objects on, or very close to, the road.  This avoids
    # treating far-away decorations as obstacles.
    nearby_road = cv2.dilate(
        road_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    )
    hazard_mask[nearby_road == 0] = 0

    return keep_object_blobs(hazard_mask)


def make_safe_mask(road_mask, hazard_mask, margin):
    """Make the pixels where the vehicle is allowed to travel."""

    # Shrinking the road keeps the route away from both road boundaries.
    road_margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * margin + 1, 2 * margin + 1)
    )
    inside_road = cv2.erode(road_mask, road_margin_kernel)

    # The road itself is already shrunk above.  Add a small extra gap around
    # each hazard, without making a narrow road section impossible to use.
    hazard_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )
    enlarged_hazards = cv2.dilate(hazard_mask, hazard_kernel)

    safe_mask = cv2.bitwise_and(
        inside_road,
        cv2.bitwise_not(enlarged_hazards)
    )

    # Remove only tiny leftover dots.  Do not keep just one component here:
    # a safe route can use either side of a pothole.
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        safe_mask,
        connectivity=8
    )
    clean_mask = np.zeros_like(safe_mask)

    for label in range(1, number):
        if stats[label, cv2.CC_STAT_AREA] >= 50:
            clean_mask[labels == label] = 255

    return clean_mask


# ----------------------------------------------------------------
# START POINT AND CHECKPOINTS
# ----------------------------------------------------------------

def nearest_white_pixel(point, mask):
    """Move a point to the nearest white pixel in a mask."""

    x, y = point
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    distances = (xs - x) ** 2 + (ys - y) ** 2
    nearest_index = np.argmin(distances)

    return int(xs[nearest_index]), int(ys[nearest_index])


def find_start_point(image, safe_mask):
    """Find the white START arrow and snap its tip to the safe road."""

    # START text and the arrow are bright grey/white.  The other light
    # squares in the picture are darker than this.
    bright = np.all(image >= 205, axis=2).astype(np.uint8) * 255

    # In a few images the arrow head and arrow line have a tiny gap.
    bright = cv2.dilate(bright, np.ones((7, 7), np.uint8), iterations=1)

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8
    )

    if number <= 1:
        return nearest_white_pixel((image.shape[1] // 2, image.shape[0] // 2), safe_mask)

    # The arrow is much taller/larger than one letter in the word START.
    arrow_label = max(
        range(1, number),
        key=lambda label: (
            int(stats[label, cv2.CC_STAT_AREA])
            + 10 * int(stats[label, cv2.CC_STAT_HEIGHT])
        )
    )

    arrow_mask = np.zeros_like(bright)
    arrow_mask[labels == arrow_label] = 255

    # The arrowhead is thicker than the thin arrow line.  The pixel deepest
    # inside this white shape is therefore a good arrowhead/start position.
    arrow_thickness = cv2.distanceTransform(
        arrow_mask,
        cv2.DIST_L2,
        5
    )
    tip_y, tip_x = np.unravel_index(
        np.argmax(arrow_thickness),
        arrow_thickness.shape
    )

    arrow_tip = (int(tip_x), int(tip_y))

    return nearest_white_pixel(arrow_tip, safe_mask)


def find_inner_hole(road_mask):
    """Find the large empty area enclosed by the looped road."""

    inverted_road = cv2.bitwise_not(road_mask)
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted_road,
        connectivity=8
    )

    image_height, image_width = road_mask.shape
    best_label = None
    best_area = 0

    for label in range(1, number):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        # The outside background touches an image edge.  The course's inner
        # empty area does not, so this removes the outside background.
        touches_edge = (
            x == 0
            or y == 0
            or x + width >= image_width
            or y + height >= image_height
        )

        if not touches_edge and area > best_area:
            best_label = label
            best_area = area

    if best_label is None:
        return None

    hole_mask = np.zeros_like(road_mask)
    hole_mask[labels == best_label] = 255

    return hole_mask


def points_along_contour(contour, count):
    """Return equally spaced points around one closed contour."""

    contour_points = contour[:, 0, :].astype(np.float32)
    next_points = np.vstack((contour_points[1:], contour_points[:1]))

    segment_lengths = np.linalg.norm(next_points - contour_points, axis=1)
    total_length = float(np.sum(segment_lengths))

    if total_length == 0:
        return []

    cumulative_length = np.cumsum(segment_lengths)
    wanted_lengths = np.linspace(0, total_length, count, endpoint=False)

    sampled_points = []

    for wanted_length in wanted_lengths:
        segment_index = int(np.searchsorted(cumulative_length, wanted_length))
        previous_length = (
            0 if segment_index == 0 else cumulative_length[segment_index - 1]
        )
        part_of_segment = (
            (wanted_length - previous_length)
            / segment_lengths[segment_index]
        )

        point = (
            contour_points[segment_index]
            + part_of_segment
            * (next_points[segment_index] - contour_points[segment_index])
        )
        sampled_points.append((int(point[0]), int(point[1])))

    return sampled_points


def move_inside_road(point, safe_mask, clearance):
    """Move an inner-boundary point towards the middle of the safe road."""

    x, y = point
    image_height, image_width = safe_mask.shape
    search_radius = 70

    left = max(0, x - search_radius)
    right = min(image_width, x + search_radius + 1)
    top = max(0, y - search_radius)
    bottom = min(image_height, y + search_radius + 1)

    local_safe = safe_mask[top:bottom, left:right] > 0
    local_ys, local_xs = np.where(local_safe)

    if len(local_xs) == 0:
        return nearest_white_pixel(point, safe_mask)

    local_xs = local_xs + left
    local_ys = local_ys + top

    distance_from_boundary = np.sqrt((local_xs - x) ** 2 + (local_ys - y) ** 2)

    # High clearance means a point is near the centre of the road.  A small
    # distance penalty makes sure that we stay near this particular contour.
    scores = clearance[local_ys, local_xs] - 0.25 * distance_from_boundary
    best_index = np.argmax(scores)

    return int(local_xs[best_index]), int(local_ys[best_index])


def create_checkpoints(road_mask, safe_mask, clearance):
    """Place ordered checkpoints around the middle of the track."""

    hole_mask = find_inner_hole(road_mask)

    if hole_mask is None:
        return []

    contours, _ = cv2.findContours(
        hole_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return []

    inner_contour = max(contours, key=cv2.contourArea)
    boundary_points = points_along_contour(inner_contour, CHECKPOINT_COUNT)

    checkpoints = []

    for point in boundary_points:
        checkpoint = move_inside_road(point, safe_mask, clearance)

        if checkpoint is None:
            continue

        # Avoid adding the same checkpoint twice.
        if not checkpoints:
            checkpoints.append(checkpoint)
            continue

        previous_x, previous_y = checkpoints[-1]
        current_x, current_y = checkpoint

        if math.hypot(current_x - previous_x, current_y - previous_y) > 12:
            checkpoints.append(checkpoint)

    return checkpoints


# ----------------------------------------------------------------
# A* PATH PLANNING
# ----------------------------------------------------------------

def make_small_grid(safe_mask, clearance):
    """Convert full image masks to the smaller grid used by A*."""

    image_height, image_width = safe_mask.shape
    grid_width = image_width // GRID_SIZE
    grid_height = image_height // GRID_SIZE

    # A grid cell is valid only when almost all of its pixels are safe.
    small_safe = cv2.resize(
        safe_mask,
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA
    )
    valid_grid = small_safe >= 240

    small_clearance = cv2.resize(
        clearance,
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA
    ) / GRID_SIZE

    return valid_grid, small_clearance


def nearest_valid_grid_point(point, valid_grid):
    """Snap an image point to its nearest valid A* grid cell."""

    x, y = point
    grid_height, grid_width = valid_grid.shape

    grid_x = min(grid_width - 1, max(0, int(x // GRID_SIZE)))
    grid_y = min(grid_height - 1, max(0, int(y // GRID_SIZE)))

    if valid_grid[grid_y, grid_x]:
        return grid_x, grid_y

    ys, xs = np.where(valid_grid)

    if len(xs) == 0:
        return None

    distances = (xs - grid_x) ** 2 + (ys - grid_y) ** 2
    closest_index = np.argmin(distances)

    return int(xs[closest_index]), int(ys[closest_index])


def astar(valid_grid, clearance_grid, start, goal):
    """Find one safe path between two grid cells using 8-direction A*."""

    if start is None or goal is None:
        return None

    if start == goal:
        return [start]

    grid_height, grid_width = valid_grid.shape

    # g_score is the known cost from start.  parent stores the previous
    # cell, so we can rebuild the final route once the goal is reached.
    g_score = np.full((grid_height, grid_width), np.inf, dtype=np.float32)
    parent_x = np.full((grid_height, grid_width), -1, dtype=np.int32)
    parent_y = np.full((grid_height, grid_width), -1, dtype=np.int32)

    start_x, start_y = start
    goal_x, goal_y = goal

    g_score[start_y, start_x] = 0
    open_list = [(0, 0, start_x, start_y)]

    neighbours = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1)
    ]

    while open_list:
        _, current_cost, current_x, current_y = heapq.heappop(open_list)

        # Ignore an older queue item if a cheaper path was found later.
        if current_cost > g_score[current_y, current_x]:
            continue

        if (current_x, current_y) == goal:
            path = []
            x, y = goal

            while (x, y) != start:
                path.append((x, y))
                old_x = parent_x[y, x]
                old_y = parent_y[y, x]
                x, y = int(old_x), int(old_y)

            path.append(start)
            path.reverse()

            return path

        for change_x, change_y in neighbours:
            next_x = current_x + change_x
            next_y = current_y + change_y

            if (
                next_x < 0
                or next_y < 0
                or next_x >= grid_width
                or next_y >= grid_height
                or not valid_grid[next_y, next_x]
            ):
                continue

            # Do not cut diagonally through the corner of an unsafe cell.
            if change_x != 0 and change_y != 0:
                if (
                    not valid_grid[current_y, next_x]
                    or not valid_grid[next_y, current_x]
                ):
                    continue

            move_cost = math.hypot(change_x, change_y)

            # Moving close to a boundary or hazard costs a little more, so
            # A* naturally prefers the middle of the road when possible.
            safety_cost = 1.0 + 2.0 / (clearance_grid[next_y, next_x] + 1.0)
            new_cost = current_cost + move_cost * safety_cost

            if new_cost < g_score[next_y, next_x]:
                g_score[next_y, next_x] = new_cost
                parent_x[next_y, next_x] = current_x
                parent_y[next_y, next_x] = current_y

                remaining_distance = math.hypot(
                    goal_x - next_x,
                    goal_y - next_y
                )
                estimated_total = new_cost + remaining_distance

                heapq.heappush(
                    open_list,
                    (estimated_total, new_cost, next_x, next_y)
                )

    # There is no connection between start and goal in this safe map.
    return None


def plan_one_loop(road_mask, safe_mask, start_point):
    """Create a closed A* route through checkpoints around the track."""

    clearance = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 5)
    checkpoints = create_checkpoints(road_mask, safe_mask, clearance)

    if start_point is None or len(checkpoints) < 6:
        return None

    # Start at the checkpoint nearest to the START arrow.  The list is then
    # rotated, which makes the route begin at START and still go around once.
    start_x, start_y = start_point
    nearest_checkpoint = min(
        range(len(checkpoints)),
        key=lambda index: (
            (checkpoints[index][0] - start_x) ** 2
            + (checkpoints[index][1] - start_y) ** 2
        )
    )

    ordered_checkpoints = (
        checkpoints[nearest_checkpoint + 1:]
        + checkpoints[:nearest_checkpoint + 1]
    )

    valid_grid, clearance_grid = make_small_grid(safe_mask, clearance)
    current = nearest_valid_grid_point(start_point, valid_grid)

    if current is None:
        return None

    complete_path = []

    # Go from START through all checkpoints and finally back to START.
    targets = ordered_checkpoints + [start_point]

    for target in targets:
        goal = nearest_valid_grid_point(target, valid_grid)
        part_path = astar(valid_grid, clearance_grid, current, goal)

        if part_path is None:
            return None

        if complete_path:
            complete_path.extend(part_path[1:])
        else:
            complete_path.extend(part_path)

        current = goal

    return complete_path


# ----------------------------------------------------------------
# DRAW AND SAVE
# ----------------------------------------------------------------

def draw_path(image, grid_path):
    """Draw the grid route on the original full-size image."""

    output = image.copy()

    image_height, image_width = image.shape[:2]
    path_points = []

    for grid_x, grid_y in grid_path:
        x = min(image_width - 1, grid_x * GRID_SIZE + GRID_SIZE // 2)
        y = min(image_height - 1, grid_y * GRID_SIZE + GRID_SIZE // 2)
        path_points.append([x, y])

    path_points = np.array(path_points, dtype=np.int32).reshape((-1, 1, 2))

    # A thin black outline makes the green path visible on all road shades.
    cv2.polylines(output, [path_points], False, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.polylines(output, [path_points], False, (0, 255, 0), 5, cv2.LINE_AA)

    start_x, start_y = path_points[0, 0]
    cv2.circle(output, (int(start_x), int(start_y)), 9, (0, 0, 0), -1)
    cv2.circle(output, (int(start_x), int(start_y)), 6, (0, 255, 0), -1)

    cv2.putText(
        output,
        "Safe A* path",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return output


def process_image(filename):
    """Plan and save a complete loop for one input image."""

    input_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(input_path)

    if image is None:
        print("Could not read:", filename)
        return

    road_mask = detect_road(image)

    if cv2.countNonZero(road_mask) == 0:
        print("Road was not found in:", filename)
        return

    hazard_mask = detect_hazards(image, road_mask)
    grid_path = None

    # Try a comfortable safety margin first.  If a narrow part of a track is
    # blocked, make the margin slightly smaller while still staying on road.
    for margin in (5, 3, 1):
        safe_mask = make_safe_mask(road_mask, hazard_mask, margin)
        start_point = find_start_point(image, safe_mask)
        grid_path = plan_one_loop(road_mask, safe_mask, start_point)

        if grid_path is not None:
            print("Used safety margin:", margin, "pixels")
            break

    if grid_path is None:
        print("Could not find a complete safe loop for:", filename)
        return

    output = draw_path(image, grid_path)

    name, extension = os.path.splitext(filename)
    output_name = name + "_safe_path" + extension
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    cv2.imwrite(output_path, output)

    print("Saved:", output_name)


def main():
    image_files = [
        filename
        for filename in os.listdir(INPUT_FOLDER)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Sort 1, 2, 3 ... 10 instead of 1, 10, 2 ...
    image_files.sort(key=lambda filename: int(os.path.splitext(filename)[0]))

    for filename in image_files:
        print("\nProcessing:", filename)
        process_image(filename)

    print("\nTask 4 completed")


if __name__ == "__main__":
    main()