import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import cv2


def toBinary(img):
    '''
    Transform an image so every pixel that is not black becomes white
    :param img: An image
    :return: A copy of the original image, with all the pixel that are not black have 1.0
    '''
    gray = img[:, :, 2] + img[:, :, 1] + img[:, :, 0]
    binary = np.zeros(gray.shape, dtype=np.float32)
    binary[gray > 0] = 1.0
    return binary


def fit_polynomial(warped, left, right):
    '''
    :param binary_warped: a binary image tha represent a bird-eye view of the street
    :return: the lane lines(left and right) that have been detected,
    and an image that can be visualized for debugging purposes
    '''
    binary_warped = toBinary(warped)

    if (not left.detected):
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    else:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels_from_prior(binary_warped, left, right)

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_fit = (np.polyfit(lefty, leftx, 2))
        left.update_fitted(left_fit, ploty)
    except Exception:
        left.detected = False

    try:
        right_fit = (np.polyfit(righty, rightx, 2))
        right.update_fitted(right_fit, ploty)
    except Exception:
        right.detected = False

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return left, right, out_img


def find_lane_pixels(binary_warped):
    '''
    :param binary_warped: a binary image tha represent a bird-eye view of the street
    :return: the pixels that are likely to belong to lane lines
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # when no pixels are found in a window, add an artificial point on the center of the window,
    # that will be used for fitting the polynom
    artificial_leftx = []
    artificial_lefty = []
    artificial_rightx = []
    artificial_righty = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image

        try:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
        except Exception:
            None

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            # otherwise fit a point located on the center of the window
            artificial_leftx.append(int(leftx_current))
            artificial_lefty.append(int((win_y_high + win_y_low) / 2))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            artificial_rightx.append(int(rightx_current))
            artificial_righty.append(int((win_y_high + win_y_low) / 2))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    leftx = np.append(leftx, np.array(artificial_leftx, dtype=np.int32))
    lefty = np.append(lefty, np.array(artificial_lefty, dtype=np.int32))
    rightx = np.append(rightx, np.array(artificial_rightx, dtype=np.int32))
    righty = np.append(righty, np.array(artificial_righty, dtype=np.int32))

    return leftx, lefty, rightx, righty, out_img


def find_lane_pixels_from_prior(binary_warped, left, right):
    margin = 100

    mask_left = np.zeros_like(binary_warped)
    mask_right = np.zeros_like(binary_warped)

    for y in left.ploty:
        mask_left[int(y), int(left.best_plotx[int(y)]) - margin:int(left.best_plotx[int(y)]) + margin] = 1.0
        mask_right[int(y), int(right.best_plotx[int(y)]) - margin:int(right.best_plotx[int(y)]) + margin] = 1.0

    pixels_left = binary_warped * mask_left
    nonzero_left = pixels_left.nonzero()
    nonzerox_left = np.array(nonzero_left[1])
    nonzeroy_left = np.array(nonzero_left[0])

    pixels_right = binary_warped * mask_right
    nonzero_right = pixels_right.nonzero()
    nonzerox_right = np.array(nonzero_right[1])
    nonzeroy_right = np.array(nonzero_right[0])

    masks = (mask_left + mask_right) * 255
    out_img = np.dstack((masks, masks, masks))

    return nonzerox_left, nonzeroy_left, nonzerox_right, nonzeroy_right, out_img


# convolution
def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids
