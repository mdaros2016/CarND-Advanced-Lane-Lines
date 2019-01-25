import numpy as np


class Line():
    '''
    Data about a detected lane line
    '''
    def __init__(self):
        # how many detections we want to store
        self.history = 25

        # was the line detected in the last iteration?
        self.detected = False

        # y values for line pixels
        self.ploty = None

        # x values for line pixels
        self.current_plotx = [np.array([])]

        # x values of the last n fits of the line
        self.recent_plotx = np.array([])

        # average x values of the fitted line over the last n iterations
        self.best_plotx = None

        # polynomial coefficients for the most recent fit
        self.current_fit = None

        # polynomial coefficients for the last fits
        self.recent_fit = np.array([])

        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None

    def update_fitted(self, current_fit, ploty):
        '''
        stores information about a new polynom that have been fit to the lane
        :param current_fit: the coefficients of the 2nd grade polynom that we want to store
        :param ploty: the coordinates of all the y points of the image
        :param count: count of the pixels that have been fitted to the polynom
        :return:
        '''
        self.ploty = ploty
        self.current_fit = current_fit

        self.current_plotx = current_fit[0] * ploty ** 2 + current_fit[1] * ploty + current_fit[2]

        self.recent_plotx = np.append(self.current_plotx, self.recent_plotx).reshape(-1, 720)
        self.recent_fit = np.append(current_fit, self.recent_fit).reshape(-1, 3)

        if (self.recent_plotx.shape[1]) >= self.history:
            self.recent_plotx = self.recent_plotx[0:self.history]
            self.recent_fit = self.recent_fit[0:self.history]

        self.best_plotx = np.mean(self.recent_plotx, 0).astype(int)

        self.detected = True
