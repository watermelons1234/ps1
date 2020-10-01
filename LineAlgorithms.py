import cv2
from matplotlib import pyplot as plt
import numpy as np


def hough_lines_acc(image_edges, theta_range=np.arange(0, 179, 1), rho_resolution=200):
    max_dim = max(image_edges.shape)
    H = np.zeros((rho_resolution, theta_range.shape[0]))
    edge_points = np.argwhere(image_edges > 0)
    rho_range = np.linspace(-2 * max_dim, 2 * max_dim, rho_resolution)
    for point in edge_points:
        x = point[1]
        y = point[0]
        for i in range(theta_range.shape[0]):
            angle = theta_range[i] * np.pi / 180
            rho = x * np.cos(angle) + y * np.sin(angle)
            rho_idx = np.abs(rho_range - rho).argmin()
            H[rho_idx, i] += 1
    return H, rho_range, theta_range


def display_hough_heatmap(H):
    H *= (255 / np.max(H))
    H = H.astype('uint8')
    plt.imshow(H, cmap='gray')
    plt.show()


def in_neighborhood(position, taken_positions, row_neigh_size, col_neigh_size, H_shape):
    position_unrav = np.unravel_index(position, H_shape)
    for taken_pos in taken_positions:
        taken_pos_unrav = np.unravel_index(taken_pos, H_shape)
        taken_i = taken_pos_unrav[0]
        taken_j = taken_pos_unrav[1]
        taken_neighborhood_rows = [taken_i + e for e in range(-row_neigh_size, row_neigh_size + 1)]
        taken_neighborhood_cols = [taken_j + e for e in range(-col_neigh_size, col_neigh_size + 1)]
        if position_unrav[0] in taken_neighborhood_rows and position_unrav[1] in taken_neighborhood_cols:
            return True
    return False


def hough_peaks(H, num_peaks, row_neigh_size=5, col_neigh_size=1):
    sorted_positions = np.argsort(H.flatten())[::-1]
    peaks_counter = 0
    position_index = 0
    maximal_positions = []
    while peaks_counter < num_peaks:
        position = sorted_positions[position_index]
        if not in_neighborhood(position, maximal_positions, row_neigh_size, col_neigh_size, H.shape):
            maximal_positions.append(position)
            peaks_counter += 1
        position_index += 1
    maximal_positions = np.array(maximal_positions)
    maximal_is, maximal_js = np.unravel_index(maximal_positions, H.shape)
    return np.vstack((maximal_is, maximal_js)).T


def hough_lines_draw(img, peaks, rho_range, theta_range):
    for peak in peaks:
        rho = rho_range[peak[0]]
        angle = theta_range[peak[1]] * np.pi / 180
        sin = np.sin(angle)
        cos = np.cos(angle)
        draw_polar_line(img, rho, sin, cos)


def draw_polar_line(img, rho, sin, cos):
    if -0.01 < sin < 0.01:
        x = int(rho / cos)
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 200, 0), thickness=2)
        return
    if -0.01 < cos < 0.01:
        y = int(rho / sin)
        cv2.line(img, (0, y), (img.shape[1], y), (0, 200, 0), thickness=2)
        return
    y0 = 0
    x0 = int(rho / cos)
    y1 = img.shape[0]
    x1 = int((rho - y1 * sin) / cos)
    cv2.line(img, (x0, y0), (x1, y1), (0, 200, 0), thickness=2)



