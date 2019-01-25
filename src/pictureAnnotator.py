import cv2
import numpy as np




class PictureAnnotator:
    '''
    Class for annotating images with information about the detected lines, curvature radius and offset from the center of the lane
    '''

    def __init__(self, perspective_transformer):
        self.perspective_transformer = perspective_transformer

    def decorate(self, img, left, right, curvature, offset):
        '''
        Adds all the relevant information to an image
        :param img: An image
        :param left: Detected left lane line
        :param right: Detected right lane line
        :param curvature: Detected curvature radius
        :param offset: Detected distance from the center of the lane
        :return: an image decorated where the lane is highlighted in green,
        and the curvature and offset values are written as text
        '''
        mask = self.get_mask(img, left, right)
        mask_unwarped = self.perspective_transformer.to_original(mask)
        final = cv2.addWeighted(img, 1, mask_unwarped, 0.8, 1)

        self.write_curvature(curvature, final)

        self.write_offset(final, offset)

        return final

    def get_mask(self, img, left, right):
        '''
        :param img: Original image
        :param left: Left line
        :param right: Right line
        :return: a black image where the area enclosed by the left and right lane lines is green
        '''
        mask = np.zeros_like(img, dtype=np.uint8)

        for y in left.ploty:
            mask[int(y), int(left.best_plotx[int(y)]):int(right.best_plotx[int(y)])] = [0, 255, 0]
            mask[int(y), int(left.best_plotx[int(y)]):int(left.best_plotx[int(y)] + 10)] = [255, 0, 0]
            mask[int(y), int(right.best_plotx[int(y)]):int(right.best_plotx[int(y)] + 10)] = [0, 0, 255]
        return mask

    def write_offset(self, final, offset):
        '''
        Writes the offset value to an image
        :param final: An image
        :param offset:  the offset value
        :return: The image with the offset value printed as text
        '''
        if (offset > 0):
            offset_text = "{:10.2f}".format(offset) + " m right"
        else:
            if (offset < 0):
                offset_text = "{:10.2f}".format(-1 * offset) + " m left"
            else:
                offset_text = "center"
        self.write_text("position: ",
                        final,
                        (10, 80))
        self.write_text(offset_text,
                        final,
                        (100, 80))

    def write_curvature(self, curvature, final):
        '''
        Writes the curvature radius value to an image
        :param curvature: the curvature value
        :param final: An image
        :return: The image with the curvature radius printed as text
        '''
        curvature_text = "{:10.0f}".format(curvature) + " m"
        self.write_text("curvature: ",
                        final,
                        (10, 50))

        self.write_text(curvature_text,
                        final,
                        (100, 50))


    def write_text(self, text, img, position):
        cv2.putText(img,
                    text,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255))
