import cv2
from matplotlib import pyplot as plt
import numpy as np
import LineAlgorithms as LA
import CircleAlgorithms as CA


IMG_1_2 = cv2.imread('input/ps1-input0.png')
GRAY_1_2 = cv2.cvtColor(IMG_1_2, cv2.COLOR_BGR2GRAY)
EDGES_1_2 = cv2.Canny(GRAY_1_2, 100, 200)

IMG_3 = cv2.imread('input/ps1-input0-noise.png')
GRAY_3 = cv2.cvtColor(IMG_3, cv2.COLOR_BGR2GRAY)
EDGES_3 = cv2.Canny(GRAY_3, 100, 200)

IMG_4_5 = cv2.imread('input/ps1-input1.png')
GRAY_4_5 = cv2.cvtColor(IMG_4_5, cv2.COLOR_BGR2GRAY)
EDGES_4_5 = cv2.Canny(GRAY_4_5, 100, 200)

IMG_6_7 = cv2.imread('input/ps1-input2.png')
GRAY_6_7 = cv2.cvtColor(IMG_6_7, cv2.COLOR_BGR2GRAY)


def apply_hough_lines_on_image(image, image_edges, num_peaks, row_neigh_size=5, col_neigh_size=1):
    H, rho_range, theta_range = LA.hough_lines_acc(image_edges, rho_resolution=4 * max(image_edges.shape))
    peaks = LA.hough_peaks(H, num_peaks, row_neigh_size, col_neigh_size)
    LA.hough_lines_draw(image, peaks, rho_range, theta_range)
    plt.imshow(image)
    plt.show()


def apply_hough_circles_on_image(image, image_gray, image_edges, radius_range, num_peaks, neigh_shape=(5, 5, 5),
                                 angular_error=0):
    first_neigh_size, second_neigh_size, third_neigh_size = neigh_shape
    H = CA.find_circles_efficiently(image_gray, image_edges, radius_range, angular_error)
    peaks = CA.hough_peaks_3d(H, num_peaks, first_neigh_size, second_neigh_size, third_neigh_size)
    CA.draw_circle_3d_peaks(image, radius_range, peaks)
    plt.imshow(image)
    plt.show()


def close_parallel_line(peak, peaks, rho_range, theta_range, upper_distance, lower_distance):
    parallel_lines = peaks[np.where(np.abs(theta_range[peaks[:, 1]] - theta_range[peak[1]]) < 2)]
    if parallel_lines.shape[0] < 2:
        return False
    closest_parallel_line = np.sort(np.abs(rho_range[parallel_lines[:, 0]] - rho_range[peak[0]]))[1]
    return rho_range[-1] * lower_distance < closest_parallel_line < rho_range[-1] * upper_distance


def get_close_parallel_peaks(peaks, rho_range, theta_range, upper_distance=1 / 30, lower_distance=0):
    new_peaks = []
    for peak in peaks:
        if close_parallel_line(peak, peaks, rho_range, theta_range, upper_distance, lower_distance):
            new_peaks.append(peak)
    return new_peaks


def q2():
    image = IMG_1_2.copy()
    apply_hough_lines_on_image(image, EDGES_1_2, 6)


def q3():
    image = IMG_3.copy()
    image_blur = cv2.GaussianBlur(image, (7, 7), 2)
    gray_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    image_blur_edges = cv2.Canny(gray_blur, 90, 190)
    apply_hough_lines_on_image(image, image_blur_edges, 6)


def q4():
    pens_blur = cv2.GaussianBlur(IMG_4_5, (7, 7), 2)
    gray_pens_blur = cv2.cvtColor(pens_blur, cv2.COLOR_BGR2GRAY)
    pens_blur_edges = cv2.Canny(gray_pens_blur, 180, 220)
    image = IMG_4_5.copy()
    apply_hough_lines_on_image(image, pens_blur_edges, 4, 10, 2)


def q5():
    circles_blur = cv2.GaussianBlur(IMG_4_5, (7, 7), 3)
    gray_circles_blur = cv2.cvtColor(circles_blur, cv2.COLOR_BGR2GRAY)
    circles_blur_edges = cv2.Canny(gray_circles_blur, 60, 120)
    circles_copy = IMG_4_5.copy()
    apply_hough_circles_on_image(circles_copy, gray_circles_blur, circles_blur_edges, range(10, 35), 14,
                                 angular_error=2)


def q6():
    img_6_blur = cv2.GaussianBlur(IMG_6_7, (7, 7), 1)
    img_6_blur_gray = cv2.cvtColor(img_6_blur, cv2.COLOR_BGR2GRAY)
    img_6_blur_edges = cv2.Canny(img_6_blur_gray, 60, 350)
    img_6_copy = IMG_6_7.copy()
    H, rho_range, theta_range = LA.hough_lines_acc(img_6_blur_edges, rho_resolution=4 * max(img_6_blur_edges.shape))
    peaks = LA.hough_peaks(H, 12, row_neigh_size=10)
    new_peaks = get_close_parallel_peaks(peaks, rho_range, theta_range)
    LA.hough_lines_draw(img_6_copy, new_peaks, rho_range, theta_range)
    plt.imshow(img_6_copy)
    plt.show()


def q7():
    img_7_blur = cv2.GaussianBlur(IMG_6_7, (5, 5), 1)
    img_7_blur_gray = cv2.cvtColor(img_7_blur, cv2.COLOR_BGR2GRAY)
    img_7_blur_edges = cv2.Canny(img_7_blur_gray, 60, 350)
    img7_copy = IMG_6_7.copy()
    apply_hough_circles_on_image(img7_copy, img_7_blur_gray, img_7_blur_edges, range(20, 35), 14, (30, 30, 10), angular_error=5)


def q8():
    print("do I have to? it's ashkara the same as 6 and 7 and it's digging very much")


def question(number):
    switcher = {
        2: q2,
        3: q3,
        4: q4,
        5: q5,
        6: q6,
        7: q7,
        8: q8
    }
    func = switcher.get(number, lambda: "not a valid question number")
    print("working on question", number)
    func()


if __name__ == '__main__':
    for i in range(2, 9):
        question(i)

















# def q5():
#     circles_blur = cv2.GaussianBlur(IMG_4_5, (7, 7), 3)
#     gray_circles_blur = cv2.cvtColor(circles_blur, cv2.COLOR_BGR2GRAY)
#     circles_blur_edges = cv2.Canny(gray_circles_blur, 60, 120)
#     H = CA.hough_circles_acc(circles_blur_edges, 20)
#     LA.display_hough_heatmap(H)
#     peaks = LA.hough_peaks(H, 20)
#     new_img_circles = IMG_4_5.copy()
#     for peak in peaks:
#         x = peak[1]
#         y = peak[0]
#         cv2.circle(new_img_circles, (x, y), 20, (0, 200, 0), thickness=2)
#     plt.imshow(new_img_circles)
#     plt.show()
#     new_img_circles = IMG_4_5.copy()
#     H, radius_range = CA.find_circles_bruteforce(circles_blur_edges, range(18, 50))
#     for i in range(len(radius_range)):
#         cv2.circle(new_img_circles, )