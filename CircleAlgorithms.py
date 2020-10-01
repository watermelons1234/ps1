import cv2
from matplotlib import pyplot as plt
import numpy as np


def hough_circles_acc(image_edges, radius):
    H = np.zeros(image_edges.shape)
    edge_points = np.argwhere(image_edges > 0)
    for point in edge_points:
        x = point[1]
        y = point[0]
        for i in range(360):
            theta = i * np.pi / 180
            a = int(x + radius * np.cos(theta))
            b = int(y - radius * np.sin(theta))
            if a in range(image_edges.shape[0]) and b in range(image_edges.shape[1]):
                H[b, a] += 1
    return H


def find_circles_bruteforce(image_edges, radius_range):
    H = np.zeros((image_edges.shape[0], image_edges.shape[1], len(radius_range)))
    for i in range(len(radius_range)):
        r = radius_range[i]
        print("r:", r)
        H[:, :, i] = hough_circles_acc(image_edges, r)
    return H, radius_range


def in_neighborhood_3d(position, taken_positions, first_neigh_size, second_neigh_size, third_neigh_size, H_shape):
    position_unrav = np.unravel_index(position, H_shape)
    for taken_pos in taken_positions:
        taken_pos_unrav = np.unravel_index(taken_pos, H_shape)
        taken_i = taken_pos_unrav[0]
        taken_j = taken_pos_unrav[1]
        taken_k = taken_pos_unrav[2]
        taken_neigh_i = [taken_i + e for e in range(-first_neigh_size, first_neigh_size + 1)]
        taken_neigh_j = [taken_j + e for e in range(-second_neigh_size, second_neigh_size + 1)]
        taken_neigh_k = [taken_k + e for e in range(-third_neigh_size, third_neigh_size + 1)]
        pos_i = position_unrav[0]
        pos_j = position_unrav[1]
        pos_k = position_unrav[2]
        if pos_i in taken_neigh_i and pos_j in taken_neigh_j and pos_k in taken_neigh_k:
            return True
    return False


def hough_peaks_3d(H, num_peaks, first_neigh_size=5, second_neigh_size=5, third_neigh_size=5):
    sorted_positions = np.argsort(H.flatten())[::-1]
    peaks_counter = 0
    position_index = 0
    maximal_positions = []
    while peaks_counter < num_peaks:
        position = sorted_positions[position_index]
        if not in_neighborhood_3d(position, maximal_positions, first_neigh_size, second_neigh_size, third_neigh_size,
                                  H.shape):
            maximal_positions.append(position)
            peaks_counter += 1
        position_index += 1
    maximal_positions = np.array(maximal_positions)
    return np.unravel_index(maximal_positions, H.shape)


def draw_circle_3d_peaks(img, radius_range, peaks):
    maximal_bs, maximal_as, maximal_rs = peaks
    for i in range(len(maximal_bs)):
        a = maximal_as[i]
        b = maximal_bs[i]
        r = radius_range[maximal_rs[i]]
        cv2.circle(img, (a, b), r, (0, 200, 0), thickness=2)


def point_in_image(img, a, b):
    return b in range(img.shape[0]) and a in range(img.shape[1])


def compute_theta_of_grad(grad_x, grad_y):
    if grad_x == 0:
        return np.pi / 2
    return np.arctan(grad_y / grad_x)


def find_circles_efficiently(image_gray, image_edges, radius_range, angular_error=0):
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1)
    H = np.zeros((image_edges.shape[0], image_edges.shape[1], len(radius_range)))
    edge_points = np.argwhere(image_edges > 0)
    for point in edge_points:
        x = point[1]
        y = point[0]
        grad_x = sobelx[y, x]
        grad_y = -sobely[y, x]
        theta = compute_theta_of_grad(grad_x, grad_y)
        for i in range(len(radius_range)):
            radius = radius_range[i]
            for e in range(-angular_error, angular_error + 1):
                angle = theta + e * np.pi / 180
                a0 = int(x + radius * np.cos(angle))
                b0 = int(y - radius * np.sin(angle))
                a1 = int(x - radius * np.cos(angle))
                b1 = int(y + radius * np.sin(angle))
                if point_in_image(image_edges, a0, b0):
                    H[b0, a0, i] += 1
                if point_in_image(image_edges, a1, b1):
                    H[b1, a1, i] += 1
    return H






