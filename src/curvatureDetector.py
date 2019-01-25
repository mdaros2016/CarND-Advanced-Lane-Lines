import numpy as np


def measure_curvature(left, right):
    return measure_curvature_real(left, right)


# Define conversions in x and y from pixels space to meters
ym_per_pix = (40 / 720)  # meters per pixel in y dimension (in bird-eye view)
xm_per_pix = (3.7 / 600)  # meters per pixel in x dimension (in bird-eye view)


def measure_curvature_real(left, right):
    '''
    Calculates the curvature of polynomial functions in meters.
    :param left: the left line, as returned by linesDetector
    :param right: left: the left line, as returned by linesDetector
    :return: the curvature radius of the lane defined by the 2 lines at the bottom of the picture, in meters
    '''

    ploty = left.ploty

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # compute the average curvature of the last fitted lane lines
    left_curverad = computeRadiusOfDetectedLane(left, y_eval)
    right_curverad = computeRadiusOfDetectedLane(right, y_eval)

    return (left_curverad + right_curverad) / 2


def measure_offset_real(left, right, width):
    '''
    Calculates the offset of the camera from the center of the lane lines
    :param left: the left line, as returned by linesDetector
    :param right: left: the left line, as returned by linesDetector
    :param width: the width of the picture (in meters)
    :return: the distance of the center of the picture from the c
    '''
    # the x coordinate of the bottom most point of the left line
    leftcoord = left.best_plotx[left.best_plotx.shape[0] - 1]

    # the x coordinate of the bottom most point of the right line
    rightcoord = right.best_plotx[right.best_plotx.shape[0] - 1]

    # distance between the center of the image and the center of the lane
    return (rightcoord + leftcoord - width) / 2 * xm_per_pix


def computeRadiusOfDetectedLane(line, y_eval):
    '''
    Computes the curvature radius of an array of 2nd grade polyinoms
    :param recent_fit: list of polynoms, representing the last lane lines that have been fitted
    :param y_eval: y-value where we want radius of curvature
    :return: the average curvature radius in meters of the input polynoms
    '''
    # Transform the detected bird-eye lines coordinates from pixels to meters, and fit a polynomial
    fit_coeff_real_world = np.polyfit((line.ploty * ym_per_pix), (line.best_plotx * xm_per_pix), 2)

    # Calculation of R_curve (radius of curvature)
    radius = ((1 + (
                2 * fit_coeff_real_world[0] * y_eval * ym_per_pix + fit_coeff_real_world[1]) ** 2) ** 1.5) / np.absolute \
                 (2 * fit_coeff_real_world[0])

    return radius
