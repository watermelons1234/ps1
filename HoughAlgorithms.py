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
    def hough_peaks(H, num_peaks, neigh_sizes):
        sorted_positions = np.argsort(H.flatten())[::-1]
        peaks_counter = 0
        position_index = 0
        maximal_positions = []
        while peaks_counter < num_peaks:
            position = sorted_positions[position_index]
            if not HoughAlgorithms.in_neighborhood(position, maximal_positions, neigh_sizes, H.shape):
                maximal_positions.append(position)
                peaks_counter += 1
            position_index += 1
        maximal_positions = np.array(maximal_positions)
        return np.vstack(np.unravel_index(maximal_positions, H.shape)).T

    @staticmethod
    def in_neighborhood(position, taken_positions, neigh_sizes, H_shape):
        position_unrav = np.unravel_index(position, H_shape)
        for taken_pos in taken_positions:
            taken_pos_unrav = np.unravel_index(taken_pos, H_shape)
            taken_neighs = [[taken_pos_unrav[i] + e for e in range(-neigh_sizes[i], neigh_sizes[i] + 1)] for i in
                            range(len(neigh_sizes))]
            if all(position_unrav[i] in taken_neighs[i] for i in range(len(neigh_sizes))):
                return True
        return False

    @staticmethod
    def point_in_image(img, a, b):
        return b in range(img.shape[0]) and a in range(img.shape[1])

    @staticmethod
    def compute_theta_of_grad(grad_x, grad_y):
        if grad_x == 0:
            return np.pi / 2
        return np.arctan(grad_y / grad_x)

