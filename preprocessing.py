import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def get_dots(image):
    """
    Identify braille dots on image using Connected Component Analysis (CCA).
    Filters out non-braille dots based on size using the median width of detected dots.

    Args:
        image: Input image

    Returns:
        Tuple containing average dot width and filtered dot information
    """

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply OTSU thresholding + invert colors
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smaller_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image = cv2.morphologyEx(
        image, cv2.MORPH_DILATE, kernel
    )  # Eliminate surrounding borders if any
    image = cv2.morphologyEx(
        image, cv2.MORPH_DILATE, kernel
    )  # Eliminate surrounding borders if any
    image = cv2.morphologyEx(
        image, cv2.MORPH_DILATE, smaller_kernel
    )  # Eliminate surrounding borders if any
    image = cv2.morphologyEx(
        image, cv2.MORPH_ERODE, kernel
    )  # Eliminate surrounding borders if any
    image = cv2.morphologyEx(
        image, cv2.MORPH_ERODE, kernel
    )  # Eliminate surrounding borders if any
    image = cv2.morphologyEx(
        image, cv2.MORPH_ERODE, kernel
    )  # Eliminate surrounding borders if any
    image = cv2.morphologyEx(
        image, cv2.MORPH_OPEN, kernel
    )  # Eliminate surrounding borders if any
    cv2.imwrite("braille_visualization/thresholded_image.jpg", image)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    all_dots = []  # To hold dot information
    widths = []  # To compute average width

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        center_x, center_y = centroids[i]

        # Compute circularity
        contours, _ = cv2.findContours(
            (labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            if circularity > 0.7:  # Consider only circular shapes
                all_dots.append(
                    {
                        "id": i,
                        "center": [float(center_x), float(center_y)],
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": int(area),
                        "circularity": float(circularity),
                        "width": int(w),
                    }
                )
                widths.append(w)

    # Compute median width to filter out non-braille dots
    if widths:
        median_width = np.median(widths)
        lower_bound = 0.8 * median_width
        upper_bound = 1.2 * median_width

        filtered_dots = [
            dot for dot in all_dots if lower_bound <= dot["width"] <= upper_bound
        ]
        print(
            f"Filtered {len(filtered_dots)} braille dots from {len(all_dots)} total components."
        )
    else:
        filtered_dots = []
        print("No braille dots were detected in the image.")

    return median_width if widths else 0, filtered_dots


def transform_image(image, dots):
    """
    Corrects any rotation or horizontal/vertical tilt in original image based on the scanned dot positions

    Args:
                    image: Original image
                    dots: List of scanned braille dots and their coordinates

    Returns:
                    Tuple containing the rotated image with its corresponding rotated dots
    """

    # Get the centers of the dot coordinates
    centers = np.array([dot["center"] for dot in dots])

    # Detect rotation using DBSCAN
    y_coords = centers[:, 1].reshape(-1, 1)
    y_clustering = DBSCAN(eps=10, min_samples=2).fit(y_coords)

    # Create set of rows to check for rotation
    rows = {}
    for i, label in enumerate(y_clustering.labels_):
        if label >= 0:
            if label not in rows:
                rows[label] = []
            rows[label].append(dots[i])

    # Calculate different rotated angle possibilities based on rows
    angles = []
    for row_dots in rows.values():
        row_dots.sort(key=lambda d: d["center"][0])  # Sort based on x position
        first_dot = row_dots[0]["center"]
        last_dot = row_dots[-1]["center"]
        dx = last_dot[0] - first_dot[0]
        dy = last_dot[1] - first_dot[1]
        if dx != 0:
            angles.append(np.degrees(np.arctan(dy / dx)))

    # Take the median of the angles
    rotation_angle = np.median(angles) if angles else 0
    print(f"The image has been rotated by {rotation_angle:.2f} degrees")

    img_height, img_width = image.shape[:2]
    img_center = (img_width // 2, img_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(img_center, float(rotation_angle), 1.0)
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (img_width, img_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Update dot coordinates
    rotated_dots = dots.copy()
    for i, dot in enumerate(rotated_dots):
        x, y = dot["center"]
        new_x = (
            rotation_matrix[0, 0] * x
            + rotation_matrix[0, 1] * y
            + rotation_matrix[0, 2]
        )
        new_y = (
            rotation_matrix[1, 0] * x
            + rotation_matrix[1, 1] * y
            + rotation_matrix[1, 2]
        )
        rotated_dots[i]["center"] = [float(new_x), float(new_y)]

    return rotated_image, dots


def segment_braille_cells(dots, dot_width):
    """
    Segments braille dots into cells

    Args:
        - dots: All dot information from CCA
        - dot_width: Average dot width of identified dots
    Returns: List of cells with its braille dot coordinates and box dimensions
    """
    # Standard braille cell spacing
    col_spacing = 2.5 * dot_width
    row_spacing = 2.5 * dot_width

    # Thresholds for cell segmentation
    standard_threshold = 1.8 * dot_width
    diagonal_threshold = 3.0 * dot_width
    vertical_cell_threshold = 4.0 * dot_width

    # Identify horizontal lines based on y-coordinates
    y_coords = sorted([dot["center"][1] for dot in dots])

    # Create array of y-coordinate differences from dot-to-dot
    y_diffs = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]

    # Group braille dots into "lines of dots" (each cell has 3)
    line_threshold = row_spacing * 0.5  # Accept some variance in y
    lines = []
    current_line = [y_coords[0]]  # Initially, consider all y coordinates

    # Loop through all y-coordinate differences, and separate (if any)
    # They're already sorted, so the moment there's a difference we know it's a new line
    for i, diff in enumerate(y_diffs):
        if diff > line_threshold:
            # New line detected
            lines.append(current_line)
            current_line = [y_coords[i + 1]]
        else:
            current_line.append(y_coords[i + 1])

    if current_line:  # Edge case
        lines.append(current_line)

    # Group lines into sets of three (for braille cells 2x3 dimension)
    line_groups = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            line_groups.append(lines[i : i + 3])
        else:
            # Handle incomplete groups at the end
            line_groups.append(lines[i:])

    print(f"Detected {len(lines)} lines and {len(line_groups)} line groups")

    # Assign each dot to its corresponding line in a map
    dot_to_line_group = {}
    for i, dot in enumerate(dots):
        y = dot["center"][1]
        for group_id, group in enumerate(line_groups):
            # Get all coordinates in the row group
            group_dots = [coord for line in group for coord in line]
            min_y = min(group_dots)
            max_y = max(group_dots)
            # Add some padding to the group boundaries
            if min_y - line_threshold / 2 <= y <= max_y + line_threshold / 2:
                dot_to_line_group[i] = group_id
                break

    # Initialize cells list and visited set
    cells = []
    visited = set()

    # Visit each dot (greedy)
    for i in range(len(dots)):
        if i in visited:
            continue

        # Start a new cell with current dot
        current_cell = [i]
        visited.add(i)

        # Find all dots that belong to this cell
        changed = True
        while changed:
            changed = False
            for dot_idx in current_cell[:]:
                for j in range(len(dots)):
                    # Check if dot is already visited or grouped in current cell
                    if j in visited and j not in current_cell:
                        continue
                    if j in current_cell:
                        continue

                    current_center = np.array(dots[dot_idx]["center"])
                    other_center = np.array(dots[j]["center"])

                    # Check if dots are in the same line group
                    current_group = dot_to_line_group.get(dot_idx)
                    other_group = dot_to_line_group.get(j)

                    if current_group != other_group:
                        # Not in the same braille row -> IGNORE
                        continue

                    # Calculate x and y distances
                    x_diff = abs(current_center[0] - other_center[0])
                    y_diff = abs(current_center[1] - other_center[1])
                    distance = np.linalg.norm(
                        current_center - other_center
                    )  # For checking diagonals

                    # Check normal cases for same cell
                    if y_diff < standard_threshold and x_diff < col_spacing:  # Same row
                        current_cell.append(j)
                        visited.add(j)
                        changed = True
                    elif (
                        x_diff < standard_threshold and y_diff < row_spacing
                    ):  # Same column
                        current_cell.append(j)
                        visited.add(j)
                        changed = True
                    elif (
                        x_diff < col_spacing
                        and y_diff < row_spacing
                        and distance < diagonal_threshold
                    ):  # Diagonal
                        current_cell.append(j)
                        visited.add(j)
                        changed = True
                    # Special case for dots in the same column but different rows
                    elif (
                        x_diff < standard_threshold and y_diff < vertical_cell_threshold
                    ):
                        current_cell.append(j)
                        visited.add(j)
                        changed = True

        # Cell complete; add to list
        if current_cell:
            cells.append(current_cell)

    # Calculate initial "box" for each cell
    padding = 0.8 * dot_width
    cell_boxes = []

    for cell in cells:
        # Find the bounding box for the current cell
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for dot_idx in cell:
            dot = dots[dot_idx]
            x, y = dot["center"]
            w, h = dot["bbox"][2], dot["bbox"][3]

            # Update bounding box
            min_x = min(min_x, x - w / 2)
            min_y = min(min_y, y - h / 2)
            max_x = max(max_x, x + w / 2)
            max_y = max(max_y, y + h / 2)

        # Add padding to the bounding box
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = max_x + padding
        max_y = max_y + padding

        # Calculate width and height
        width = max_x - min_x
        height = max_y - min_y

        cell_boxes.append(
            {
                "x1": min_x,
                "y1": min_y,
                "x2": max_x,
                "y2": max_y,
                "width": width,
                "height": height,
            }
        )

    # Calculate cell centroids
    cell_centroids = []
    for cell in cells:
        cell_centers = [dots[i]["center"] for i in cell]
        centroid_x = sum(center[0] for center in cell_centers) / len(cell_centers)
        centroid_y = sum(center[1] for center in cell_centers) / len(cell_centers)
        cell_centroids.append((centroid_x, centroid_y))

    # Sort cells based on their centroids
    sorted_indices = sorted(
        range(len(cells)), key=lambda i: (cell_centroids[i][1], cell_centroids[i][0])
    )
    sorted_cells = [cells[i] for i in sorted_indices]
    sorted_boxes = [cell_boxes[i] for i in sorted_indices]
    sorted_centroids = [cell_centroids[i] for i in sorted_indices]

    # Map each cell to its row (line group)
    cell_to_row = {}
    for i, cell in enumerate(sorted_cells):
        if not cell:
            continue

        # Get the first dot in the cell
        first_dot = cell[0]
        row_idx = dot_to_line_group.get(first_dot)
        if row_idx is not None:  # Shouldn't be None
            cell_to_row[i] = row_idx

    # Group cells by rows
    rows = [
        [] for _ in range(len(line_groups))
    ]  # Create empty array of arrays for each row
    for i, cell in enumerate(sorted_cells):
        if i in cell_to_row:
            rows[cell_to_row[i]].append(i)

    # Sort cells within each row by x-coordinate - should already be sorted
    for row in rows:
        row.sort(key=lambda cell_idx: sorted_centroids[cell_idx][0])

    # Standardize box heights within each row for consistncy
    for row_idx, row_cells in enumerate(rows):
        if not row_cells:
            continue

        # Find min and max y-coordinates for the row
        row_min_y = float("inf")
        row_max_y = float("-inf")

        for cell_idx in row_cells:
            row_min_y = min(row_min_y, sorted_boxes[cell_idx]["y1"])
            row_max_y = max(row_max_y, sorted_boxes[cell_idx]["y2"])

        # Update all boxes in this row to have the same height
        for cell_idx in row_cells:
            sorted_boxes[cell_idx]["y1"] = row_min_y
            sorted_boxes[cell_idx]["y2"] = row_max_y
            sorted_boxes[cell_idx]["height"] = row_max_y - row_min_y

        # Find the maximum width of all cells
        max_width = max(box["width"] for box in sorted_boxes)
        width_threshold = 0.75 * max_width  # Threshold for identifying narrow cells
        # If the width is below this, it means it's a "narrow" cell

        # Process each cell in the row
        for i, cell_idx in enumerate(row_cells):
            current_box = sorted_boxes[cell_idx]

            # Check if this is a narrow cell that needs width standardization
            if current_box["width"] < width_threshold:
                # Calculate how much to extend
                width_to_add = 0.85 * max_width - current_box["width"]

                # Try to fill up the space to the right first
                can_extend_right = True

                # Check if overlapping with the next cell on the right
                if i < len(row_cells) - 1:
                    next_cell_idx = row_cells[i + 1]
                    next_box = sorted_boxes[next_cell_idx]

                    if current_box["x2"] + width_to_add > next_box["x1"]:
                        # Overlaps! Need to extend left instead
                        can_extend_right = False

                if can_extend_right:
                    # Extend to the right
                    current_box["x2"] += width_to_add + max_width * 0.1
                else:
                    # Extend to the left
                    current_box["x1"] -= width_to_add + max_width * 0.1

                # Update width
                current_box["width"] = current_box["x2"] - current_box["x1"]

    # Create cells with combined information
    # Create a new ordered list of cells
    ordered_cells = []
    for row_idx, row in enumerate(rows):
        for cell_idx in row:
            ordered_cells.append(
                {
                    "dot_indices": sorted_cells[cell_idx],
                    "box": sorted_boxes[cell_idx],
                    "centroid": sorted_centroids[cell_idx],
                    "row": row_idx,
                }
            )

    # Print the number of rows
    print(f"Number of rows: {len(rows)}")

    # Print the cells that belong to each row
    for row_idx, row_cells in enumerate(rows):
        print(f"Row {row_idx + 1}: Cells {row_cells}")

    return ordered_cells


def classify_braille_cell(cell_info, dots, cell_index):
    """
    Classifies braille cell and exports image in sequence

    Args:
                    cell_info: List of cells and their dictionary of dot info
                    dots: list of dots
                    cell_index: What part of the sequence it's on

    Returns: The classified braille character
    """

    braille_dict = {
        # Letters
        (1, 0, 0, 0, 0, 0): "a",
        (1, 1, 0, 0, 0, 0): "b",
        (1, 0, 0, 1, 0, 0): "c",
        (1, 0, 0, 1, 1, 0): "d",
        (1, 0, 0, 0, 1, 0): "e",
        (1, 1, 0, 1, 0, 0): "f",
        (1, 1, 0, 1, 1, 0): "g",
        (1, 1, 0, 0, 1, 0): "h",
        (0, 1, 0, 1, 0, 0): "i",
        (0, 1, 0, 1, 1, 0): "j",
        (1, 0, 1, 0, 0, 0): "k",
        (1, 1, 1, 0, 0, 0): "l",
        (1, 0, 1, 1, 0, 0): "m",
        (1, 0, 1, 1, 1, 0): "n",
        (1, 0, 1, 0, 1, 0): "o",
        (1, 1, 1, 1, 0, 0): "p",
        (1, 1, 1, 1, 1, 0): "q",
        (1, 1, 1, 0, 1, 0): "r",
        (0, 1, 1, 1, 0, 0): "s",
        (0, 1, 1, 1, 1, 0): "t",
        (1, 0, 1, 0, 0, 1): "u",
        (1, 1, 1, 0, 0, 1): "v",
        (0, 1, 0, 1, 1, 1): "w",
        (1, 0, 1, 1, 0, 1): "x",
        (1, 0, 1, 1, 1, 1): "y",
        (1, 0, 1, 0, 1, 1): "z",
        (0, 0, 1, 1, 1, 1): "_",  # SIGNIFIES NUMBER
    }

    box = cell_info["box"]
    x1, x2 = box["x1"], box["x2"]
    y1, y2 = box["y1"], box["y2"]
    width, height = box["width"], box["height"]

    # Get dot indices safely
    dot_indices = cell_info.get("dot_indices", [])

    # Define braille cell positions (top-to-bottom -> left-to-right)
    positions = [
        (0.25, 0.25),  # top-left
        (0.25, 0.5),  # middle-left
        (0.25, 0.75),  # bottom-left
        (0.75, 0.25),  # top-right
        (0.75, 0.5),  # middle-right
        (0.75, 0.75),  # bottom-right
    ]

    # Initialize as all 0s
    pattern = [0] * 6

    # Assign dots to positions
    for dot_idx in dot_indices:
        dot = dots[dot_idx]
        dot_x, dot_y = dot["center"]

        # Convert to relative position within the cell
        rel_x = (dot_x - x1) / width
        rel_y = (dot_y - y1) / height

        # Find closest braille position
        closest_pos = min(
            range(6),
            key=lambda i: (rel_x - positions[i][0]) ** 2
            + (rel_y - positions[i][1]) ** 2,
        )
        pattern[closest_pos] = 1

    # Convert list to tuple and classify
    char = braille_dict.get(tuple(pattern[:6]), "?")  # Ensures exactly 6 elements

    return char


def crop_cells(image, cells, dots, output_dir="./Sequence"):
    """
    Crops each braille cell from the image, resize to 28x28, and prints it

    Args:
        image: Original image
        cells: List of cells with its dots and box information
        save_dir: Optional directory to save the cropped images

    Returns:
        List of 28x28 cropped cell images
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cropped_cells = []
    classifications = []

    width = len(str(len(cells)))  # Get width for formatting

    # Process each cell
    for i, cell_info in enumerate(cells):
        box = cell_info["box"]

        # Get box coordinates
        x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
        x2, y2 = min(image.shape[1], int(box["x2"])), min(
            image.shape[0], int(box["y2"])
        )

        # Crop the cell from the image
        cropped = image[y1:y2, x1:x2]

        # Resize to 28x28
        resized = cv2.resize(cropped, (28, 28))

        char = classify_braille_cell(cell_info, dots, i)
        classifications.append(char)

        index = f"{i:0{width}d}"
        filename = f"{output_dir}/{index}{char}.jpg"
        cv2.imwrite(filename, resized)
        cropped_cells.append(resized)

    print(f"Processed {len(cropped_cells)} cells")


def preprocessing(
    image_path="./Test Images/test.jpg", output_dir="braille_visualization"
):
    """
    Preprocess and visualize input image with standardized braille cells

    Args:
                    image_path: Path to input image
                    output_dir: Path for output visualizations
    """

    # Clear ./Sequence directory
    clear_sequence()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Failed to read image")

    # Save original image
    original_path = output_dir / "original.jpg"
    cv2.imwrite(str(original_path), original_image)

    # Perform CCA to extract dots from image
    dot_width, dots = get_dots(original_image)

    # Correct image rotation
    image, dots = transform_image(original_image, dots)

    # Save transformed image
    rotated_path = output_dir / "rotated.jpg"
    cv2.imwrite(str(rotated_path), image)

    visualize_dots = image.copy()
    for dot in dots:
        # Draw green dots around rotated image
        center = tuple(int(x) for x in dot["center"])
        radius = int(np.sqrt(dot["area"] / np.pi))
        cv2.circle(visualize_dots, center, radius, (0, 255, 0), 2)
        cv2.putText(
            visualize_dots,
            str(dot["id"]),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
    # Save visualized dots
    dots_path = output_dir / "visualize_dots.jpg"
    cv2.imwrite(str(dots_path), visualize_dots)

    # Segment dots into braille cells with standardized boxes
    cells = segment_braille_cells(dots, dot_width)
    print(f"Detected {len(cells)} braille cells")

    # Create visualization for standardized boxes
    visualize_standardized = image.copy()
    # Generate random colors for cells
    colors = [
        (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        for _ in range(len(cells))
    ]

    # Draw each cell with its standardized box
    for i, cell_info in enumerate(cells):
        box = cell_info["box"]
        dot_indices = cell_info["dot_indices"]

        # Draw standardized bounding box
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])

        cv2.rectangle(visualize_standardized, (x1, y1), (x2, y2), colors[i], 2)

        # Add cell number
        cv2.putText(
            visualize_standardized,
            f"Cell {i}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[i],
            2,
        )

        # Draw dots in the cell
        for dot_idx in dot_indices:
            center = tuple(int(x) for x in dots[dot_idx]["center"])
            cv2.circle(visualize_standardized, center, 3, colors[i], -1)
            # Display the dot id
            cv2.putText(
                visualize_standardized,
                str(dots[dot_idx]["id"]),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    # Save visualized standardized cells
    standardized_path = output_dir / "visualize_standardized_cells.jpg"
    cv2.imwrite(str(standardized_path), visualize_standardized)

    # Crop cells (we don't really do anything with returned value)
    cropped_cells = crop_cells(image, cells, dots)


def clear_sequence(folder_path="./Sequence"):
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        for file in folder.iterdir():
            file.unlink()  # Remove file


class LoadProcessedImages(Dataset):
    def __init__(self, image_folder):
        self.image_folder = Path(image_folder)
        self.image_files = list(self.image_folder.iterdir())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("L")  # Convert to grayscale
        transform = transforms.Compose([transforms.PILToTensor()])
        image = transform(image)
        label = img_name.stem[-1]

        return image, label


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
if __name__ == "__main__":
    preprocessing("./Test Images/test5.jpg")
    dataset = LoadProcessedImages("./Sequence")
    data_loader = DataLoader(dataset)

    print("\nLoading Data Loader...")

    full_string = ""  # Initialize an empty string
    for _, labels in data_loader:
        full_string += "".join(labels)

    print(full_string)
    """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
