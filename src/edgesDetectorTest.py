from unittest import TestCase

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from edgesDetector import EdgesDetector


class EdgesDetectorIntegrationTest(TestCase):
    '''
    Supports manual testing of EdgesDetector
    '''
    def test_edges_detection(self):
        '''
        cameraCalibratorTest must run before
        Detects the edges in  the picture ../test_images/bird_eye_on_straight_line.jpg (created by cameraCalibratorTest)
        and saves it to ../output_images/edges.jpg
        '''
        edgesDetector = EdgesDetector()
        original = mpimg.imread('../output_images/straight_lines1_undistorted.jpg')
        edges = edgesDetector.detectEdges(original)
        mpimg.imsave('../output_images/edges.jpg', edges * 255)
        plt.imshow(edges)
        plt.show()
