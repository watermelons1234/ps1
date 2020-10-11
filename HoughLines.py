import cv2
from HoughAlgorithms import HoughAlgorithms
from matplotlib import pyplot as plt
import numpy as np


class HoughLines(HoughAlgorithms):

    @staticmethod
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


    @staticmethod
    def hough_lines_draw(img, peaks, rho_range, theta_range):
        for peak in peaks:
            rho = rho_range[peak[0]]
            angle = theta_range[peak[1]] * np.pi / 180
            sin = np.sin(angle)
            cos = np.cos(angle)
            HoughLines.draw_polar_line(img, rho, sin, cos)

    @staticmethod
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

    @staticmethod
    def apply_hough_lines_on_image(image, image_edges, num_peaks, neigh_shape=(5,1)):
        H, rho_range, theta_range = HoughLines.hough_lines_acc(image_edges, rho_resolution=4 * max(image_edges.shape))
        peaks = HoughLines.hough_peaks(H, num_peaks, neigh_shape)
        HoughLines.hough_lines_draw(image, peaks, rho_range, theta_range)
        plt.imshow(image)
        plt.show()

    @staticmethod
    def close_parallel_line(peak, peaks, rho_range, theta_range, upper_distance, lower_distance):
        parallel_lines = peaks[np.where(np.abs(theta_range[peaks[:, 1]] - theta_range[peak[1]]) < 2)]
        if parallel_lines.shape[0] < 2:
            return False
        closest_parallel_line = np.sort(np.abs(rho_range[parallel_lines[:, 0]] - rho_range[peak[0]]))[1]
        return rho_range[-1] * lower_distance < closest_parallel_line < rho_range[-1] * upper_distance

    @staticmethod
    def get_close_parallel_peaks(peaks, rho_range, theta_range, upper_distance=1 / 30, lower_distance=0):
        new_peaks = []
        for peak in peaks:
            if HoughLines.close_parallel_line(peak, peaks, rho_range, theta_range, upper_distance, lower_distance):
                new_peaks.append(peak)
        return new_peaks

