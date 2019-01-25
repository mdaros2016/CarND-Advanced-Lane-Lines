import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import cv2


class EdgesDetector:
    '''
    Class for detecting edges in a picture, considering the saturation and the horizontal luminosity gradient
    '''
    def __init__(self, s_thresh=(210, 255), sx_thresh=(80, 220)):
        '''
        :param s_thresh: saturation threshold
        :param sx_thresh: horizontal luminosity gradient threshold
        '''
        self.s_yellow_thresh = s_thresh
        self.sx_thresh = sx_thresh
        None

    def detectEdges(self, img):
        '''
        Detects the edges of an image
        :param img: An image
        :return: a binary image where the detected corners are white, and the other pixels black
        '''
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        l_channel, s_channel = self.extract_channels(img)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sxbinary = self.horizontal_gradient_binary_mask(gray)
        s_binary = self.saturation_binary_mask(s_channel)

        # Stack each channel
        edges_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # only keep the edges that fit in a trapezoid with the base on the bottom of the picture and the short
        # side at 40% height
        xsize = img.shape[1]
        ysize = img.shape[0]
        boundaries_perspective = np.array(
            [[(xsize * 0.50, ysize * 0.55), (xsize * 0.05, ysize), (xsize * 0.95, ysize)]],
            dtype=np.int32)
        edges_cropped = self.region_of_interest(edges_binary, boundaries_perspective)
        boundaries_close = np.array([[(0, ysize), (xsize, ysize), (xsize, ysize * 0.6), (0, ysize * 0.6)]],
                                    dtype=np.int32)
        edges_cropped = self.region_of_interest(edges_cropped, boundaries_close)

        return self.toBinary(edges_cropped)

    def toBinary(self, img):
        '''
        Transform an image so every pixel that is not black becomes white
        :param img: An image
        :return: A copy of the original image, with all the pixel that are not black are turned into white
        '''
        gray = img[:, :, 2] + img[:, :, 1] + img[:, :, 0]
        binary = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
        binary[gray > 0] = [255, 255, 255]
        return binary

    def extract_channels(self, img):
        '''
        :param img: An image
        :return: 2 2D arrays, representing the L and the S channel of the original image
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        return l_channel, s_channel

    def saturation_binary_mask(self, s_channel):
        '''
        Filters out the pixels that don't fit the saturation threshold
        :param img: An image
        :return: a binary image where pixels that fit the saturation threshold are white, and the other pixels black
        '''
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[((s_channel >= self.s_yellow_thresh[0]) & (s_channel <= self.s_yellow_thresh[1]))] = 1
        return s_binary

    def horizontal_gradient_binary_mask(self, l_channel):
        '''
        Filters out the pixels that don't fit the horizontal gradient threshold
        :param img: An image
        :return: a binary image where pixels that fit the horizontal gradient threshold are white, and the other pixels black
        '''
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        # Threshold x gradient
        sxbinary = np.zeros_like(abs_sobelx)
        sxbinary[(abs_sobelx >= self.sx_thresh[0]) & (abs_sobelx <= self.sx_thresh[1])] = 1
        return sxbinary

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
