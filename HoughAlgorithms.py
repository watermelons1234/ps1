import numpy as np
import matplotlib.pyplot as plt


class HoughAlgorithms:

    @staticmethod
    def display_hough_heatmap(H):
        H *= (255 / np.max(H))
        H = H.astype('uint8')
        plt.imshow(H, cmap='gray')
        plt.show()

    @staticmethod
    def in_neighborhood_2d(position, taken_positions, row_neigh_size, col_neigh_size, H_shape):
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

    @staticmethod
    def hough_peaks_2d(H, num_peaks, row_neigh_size=5, col_neigh_size=1):
        sorted_positions = np.argsort(H.flatten())[::-1]
        peaks_counter = 0
        position_index = 0
        maximal_positions = []
        while peaks_counter < num_peaks:
            position = sorted_positions[position_index]
            if not HoughAlgorithms.in_neighborhood_2d(position, maximal_positions, row_neigh_size, col_neigh_size,
                                                      H.shape):
                maximal_positions.append(position)
                peaks_counter += 1
            position_index += 1
        maximal_positions = np.array(maximal_positions)
        maximal_is, maximal_js = np.unravel_index(maximal_positions, H.shape)
        return np.vstack((maximal_is, maximal_js)).T

    @staticmethod
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

    @staticmethod
    def hough_peaks_3d(H, num_peaks, first_neigh_size=5, second_neigh_size=5, third_neigh_size=5):
        sorted_positions = np.argsort(H.flatten())[::-1]
        peaks_counter = 0
        position_index = 0
        maximal_positions = []
        while peaks_counter < num_peaks:
            position = sorted_positions[position_index]
            if not HoughAlgorithms.in_neighborhood_3d(position, maximal_positions, first_neigh_size, second_neigh_size,
                                      third_neigh_size,
                                      H.shape):
                maximal_positions.append(position)
                peaks_counter += 1
            position_index += 1
        maximal_positions = np.array(maximal_positions)
        return np.unravel_index(maximal_positions, H.shape)

    @staticmethod
    def point_in_image(img, a, b):
        return b in range(img.shape[0]) and a in range(img.shape[1])

    @staticmethod
    def compute_theta_of_grad(grad_x, grad_y):
        if grad_x == 0:
            return np.pi / 2
        return np.arctan(grad_y / grad_x)

