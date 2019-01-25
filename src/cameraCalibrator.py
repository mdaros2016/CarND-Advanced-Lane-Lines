import glob

import cv2
import numpy as np


class CameraCalibrator:
    '''
    Class for correcting the distortion of the pictures taken from the camera.
    '''

    def __init__(self, calibration_pictures_path_pattern='../camera_cal/calibration*.jpg'):
        '''
        :param calibration_pictures_path_pattern: File system path of a set of 9x6 chessboard pictures that will be used for camera calibration
        '''
        # store mtx and dist in the status of the object, so we don't have to compute them at every iteration
        self.mtx = None
        self.dist = None
        self.calibration_pictures_path_pattern = calibration_pictures_path_pattern

    def undistort(self, img):
        '''
        Corrects the distortion of an image.
        The first invocation of thi method will take long, since it will lazily initialize the transformation matrix
        :param img: distorted picture to be corrected
        :return: the corrected picture
        '''
        if self.mtx is None:
            self.initialize_transformation_matrix()
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

    def initialize_transformation_matrix(self):
        '''
        Initializes the transformation matrix, using the pictures contained in the path specified above
        :return: Nothing, it just changes the internal status of the object
        '''
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        img_size = []

        # Make a list of calibration images
        images = glob.glob(self.calibration_pictures_path_pattern)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                append = imgpoints.append(corners)
                img_size = (img.shape[1], img.shape[0])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.mtx = mtx
        self.dist = dist
