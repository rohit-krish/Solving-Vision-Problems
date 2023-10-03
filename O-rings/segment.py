import numpy as np


def colorize_labels(labels):
    unique_labels = list(np.unique(labels))
    # unique_labels.pop(0)
    num_colors = len(unique_labels)
    colors = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)

    # Create a colored image
    colored_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        colored_image[mask] = color

    return colored_image


def _get_neighbors(x, y, shape, use_4_adjacency=True):
    # Define relative positions of neighbors; using 8-adjacency
    if use_4_adjacency:
        neighbors_indices = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        neighbors_indices = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    neighbors = []

    for dx, dy in neighbors_indices:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < shape[0] and 0 <= new_y < shape[1]:
            neighbors.append((new_x, new_y))

    return neighbors


def segment_objects(binary_image):
    binary_image_shape = binary_image.shape
    labels = np.zeros(binary_image_shape, dtype=int)
    next_label = 1

    for i in range(binary_image_shape[0]):
        for j in range(binary_image_shape[1]):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                labels[i, j] = next_label
                neighbors = _get_neighbors(i, j, binary_image_shape)
                while len(neighbors) > 0:
                    neighbor = neighbors.pop()
                    if binary_image[neighbor] == 255 and labels[neighbor] == 0:
                        labels[neighbor] = next_label
                        neighbors += _get_neighbors(
                            neighbor[0], neighbor[1], binary_image_shape
                        )
                next_label += 1
    return labels
