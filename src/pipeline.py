import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cameraCalibrator import CameraCalibrator
from perspectiveTransformer import PerspectiveTransformer
from edgesDetector import EdgesDetector
import linesDetector
from line import Line
from pictureAnnotator import PictureAnnotator
import curvatureDetector

debug = False


class Pipeline:
    '''
    Detects the lane lines on an image
    '''
    def __init__(self, camera_calibrator=CameraCalibrator(), perspective_transformer=PerspectiveTransformer(),
                 edges_detector=EdgesDetector(), picture_annotator=None):
        self.camera_calibrator = camera_calibrator
        self.perspective_transformer = perspective_transformer
        self.edges_detector = edges_detector
        if (picture_annotator is None):
            self.picture_annotator = PictureAnnotator(perspective_transformer)
        else:
            self.pictureAnnotator = picture_annotator
        self.left = Line()
        self.right = Line()

    def pipeline(self, img):
        '''
        :param img: An image representing a road with lane lines
        :return: (hopefully) the image where the most central lane is highlighted in green, and the curvature radius
        and the distance between the center of the picture and the center of the lane is printed
        '''
        undistorted = self.camera_calibrator.undistort(img)

        edges = self.edges_detector.detectEdges(undistorted)

        warped = self.perspective_transformer.to_bird_eye(edges)

        self.left, self.right, debug_img = linesDetector.fit_polynomial(warped, self.left, self.right)

        curvature = curvatureDetector.measure_curvature_real(self.left, self.right)

        offset = curvatureDetector.measure_offset_real(self.left, self.right, img.shape[1])

        final = self.picture_annotator.decorate(img, self.left, self.right, curvature, offset)

        if (debug):
            print("pipeline is in debug mode")
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(edges)
            ax1.set_title('edges', fontsize=50)
            ax2.imshow(debug_img)
            ax2.set_title('lines', fontsize=50)
            ax3.imshow(final)
            ax3.set_title('final', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

        return final
