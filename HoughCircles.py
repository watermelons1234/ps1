import cv2
from matplotlib import pyplot as plt
import numpy as np
from HoughAlgorithms import HoughAlgorithms


class HoughCircles(HoughAlgorithms):

    @staticmethod
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

    @staticmethod
    def find_circles_bruteforce(image_edges, radius_range):
        H = np.zeros((image_edges.shape[0], image_edges.shape[1], len(radius_range)))
        for i in range(len(radius_range)):
            r = radius_range[i]
            print("r:", r)
            H[:, :, i] = HoughCircles.hough_circles_acc(image_edges, r)
        return H, radius_range

    @staticmethod
    def draw_circle_3d_peaks(img, radius_range, peaks):
        maximal_bs, maximal_as, maximal_rs = peaks
        for i in range(len(maximal_bs)):
            a = maximal_as[i]
            b = maximal_bs[i]
            r = radius_range[maximal_rs[i]]
            cv2.circle(img, (a, b), r, (0, 200, 0), thickness=2)

    @staticmethod
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
            theta = HoughAlgorithms.compute_theta_of_grad(grad_x, grad_y)
            HoughCircles.mark_possible_centers_for_point(image_edges, radius_range, angular_error, theta, x, y, H)
        return H

    @staticmethod
    def mark_possible_centers_for_point(image_edges, radius_range, angular_error, theta, x, y, H):
        for i in range(len(radius_range)):
            radius = radius_range[i]
            for e in range(-angular_error, angular_error + 1):
                angle = theta + e * np.pi / 180
                a0 = int(x + radius * np.cos(angle))
                b0 = int(y - radius * np.sin(angle))
                a1 = int(x - radius * np.cos(angle))
                b1 = int(y + radius * np.sin(angle))
                if HoughAlgorithms.point_in_image(image_edges, a0, b0):
                    H[b0, a0, i] += 1
                if HoughAlgorithms.point_in_image(image_edges, a1, b1):
                    H[b1, a1, i] += 1

    @staticmethod
    def apply_hough_circles_on_image(image, image_gray, image_edges, radius_range, num_peaks, neigh_shape=(5, 5, 5),
                                     angular_error=0):
        first_neigh_size, second_neigh_size, third_neigh_size = neigh_shape
        H = HoughCircles.find_circles_efficiently(image_gray, image_edges, radius_range, angular_error)
        peaks = HoughCircles.hough_peaks_3d(H, num_peaks, first_neigh_size, second_neigh_size, third_neigh_size)
        HoughCircles.draw_circle_3d_peaks(image, radius_range, peaks)
        plt.imshow(image)
        plt.show()
