from unittest import TestCase

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cameraCalibrator import CameraCalibrator


class findCornersTest(TestCase):
    '''
    Supports manual testing of CameraCalibrator
    '''
    def testUndistort(self):
        '''
        Undistorts the picture ../test_images/straight_lines1.jpg and saves it to ../output_images/straight_lines1_undistorted.jpg
        '''
        camera_calibration = CameraCalibrator()
        original = mpimg.imread('../test_images/straight_lines1.jpg')
        undistorted = camera_calibration.undistort(original)
        mpimg.imsave('../output_images/straight_lines1_undistorted.jpg', undistorted)
        plt.imshow(undistorted)
        plt.show()
