import cv2
from matplotlib import pyplot as plt
import numpy as np
from HoughLines import HoughLines
from HoughCircles import HoughCircles
from HoughAlgorithms import HoughAlgorithms


def q2():
    img_2 = cv2.imread('input/ps1-input0.png')
    gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    HoughLines.apply_hough_lines_on_image(img_2, edges, 6)


def q3():
    img_3 = cv2.imread('input/ps1-input0-noise.png')
    image_blur = cv2.GaussianBlur(img_3, (7, 7), 2)
    gray_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    image_blur_edges = cv2.Canny(gray_blur, 90, 190)
    HoughLines.apply_hough_lines_on_image(img_3, image_blur_edges, 6)


def q4():
    img_4 = cv2.imread('input/ps1-input1.png')
    pens_blur = cv2.GaussianBlur(img_4, (7, 7), 2)
    gray_pens_blur = cv2.cvtColor(pens_blur, cv2.COLOR_BGR2GRAY)
    pens_blur_edges = cv2.Canny(gray_pens_blur, 180, 220)
    HoughLines.apply_hough_lines_on_image(img_4, pens_blur_edges, 4, 10, 2)


def q5():
    img_5 = cv2.imread('input/ps1-input1.png')
    circles_blur = cv2.GaussianBlur(img_5, (7, 7), 3)
    gray_circles_blur = cv2.cvtColor(circles_blur, cv2.COLOR_BGR2GRAY)
    circles_blur_edges = cv2.Canny(gray_circles_blur, 60, 120)
    HoughCircles.apply_hough_circles_on_image(img_5, gray_circles_blur, circles_blur_edges, range(10, 35), 14,
                                 angular_error=2)


def q6():
    img_6 = cv2.imread('input/ps1-input2.png')
    img_6_blur = cv2.GaussianBlur(img_6, (7, 7), 1)
    img_6_blur_gray = cv2.cvtColor(img_6_blur, cv2.COLOR_BGR2GRAY)
    img_6_blur_edges = cv2.Canny(img_6_blur_gray, 60, 350)
    H, rho_range, theta_range = HoughLines.hough_lines_acc(img_6_blur_edges,
                                                           rho_resolution=4 * max(img_6_blur_edges.shape))
    peaks = HoughAlgorithms.hough_peaks_2d(H, 12, row_neigh_size=10)
    new_peaks = HoughLines.get_close_parallel_peaks(peaks, rho_range, theta_range)
    HoughLines.hough_lines_draw(img_6, new_peaks, rho_range, theta_range)
    plt.imshow(img_6)
    plt.show()


def q7():
    img_7 = cv2.imread('input/ps1-input2.png')
    img_7_blur = cv2.GaussianBlur(img_7, (5, 5), 1)
    img_7_blur_gray = cv2.cvtColor(img_7_blur, cv2.COLOR_BGR2GRAY)
    img_7_blur_edges = cv2.Canny(img_7_blur_gray, 60, 350)
    HoughCircles.apply_hough_circles_on_image(img_7, img_7_blur_gray, img_7_blur_edges, range(20, 35), 14,
                                              (30, 30, 10), angular_error=5)


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