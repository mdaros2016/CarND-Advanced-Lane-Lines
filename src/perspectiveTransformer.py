import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import cv2


class PerspectiveTransformer:
    '''
    Transforms images between camera view and birt-eye view
    '''
    def __init__(self, src=[(188, 720), (1130, 720), (769, 500), (518, 500)],
                 dst=[(350, 720), (950, 720), (950, 500), (350, 500)]):
        '''
        :param src: The Coordinates of some predefined points in the camera view image
        :param dst: The coordinates of the points defined above in the bird-eye view image
        '''
        self.transform_matrix = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))

    def to_bird_eye(self, img):
        '''
        Converts an image to bird-eye view
        :param img: An image in camera view
        :return: The image transformed to bird-eye view
        '''
        return cv2.warpPerspective(img, self.transform_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def to_original(self, img):
        '''
        Converts an image to camera view
        :param img: An image in bird-eye view
        :return: The image transformed to camera view
        '''
        return cv2.warpPerspective(img, self.inverse_transform_matrix, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)
