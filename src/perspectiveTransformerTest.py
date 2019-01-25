from unittest import TestCase

import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from perspectiveTransformer import PerspectiveTransformer


class PerspectiveTransformerTest(TestCase):
    '''
      Supports manual testing of PerspectiveTransformer
      '''
    def test_perspective_transformer(self):
        '''
        cameraCalibratorTest must run before
        Transforms the image ../test_images/straight_lines1_undistorted.jpg (created by cameraCalibratorTest) to bird eye
        and saves it to ../output_images/bird_eye_on_straight_line.jpg
        '''
        original = mpimg.imread('../output_images/straight_lines1_undistorted.jpg')

        original_points = [(188, 720), (1117, 720), (769, 500), (518, 500)]
        dest_points = [(350, 720), (950, 720), (950, 500), (350, 500)]

        perspective_transformer = PerspectiveTransformer(original_points, dest_points)
        bird_eye = perspective_transformer.to_bird_eye(original)

        cv2.line(original, original_points[0], original_points[3], [255, 0, 0], 1)
        cv2.line(original, original_points[1], original_points[2], [255, 0, 0], 1)
        cv2.line(original, original_points[0], original_points[1], [255, 0, 0], 1)
        cv2.line(original, original_points[2], original_points[3], [255, 0, 0], 1)

        plt.imshow(original)
        mpimg.imsave("../output_images/undistorted_image_with_source_points_drawn.jpg", original)
        plt.show()

        plt.imshow(bird_eye)
        mpimg.imsave("../output_images/bird_eye_on_straight_line.jpg", bird_eye)
        plt.show()

        inverse = perspective_transformer.to_original(bird_eye)
        plt.imshow(inverse)
        plt.show()
