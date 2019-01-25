import os

import matplotlib.image as mpimg
from cameraCalibrator import CameraCalibrator
from moviepy.editor import VideoFileClip
from pipeline import Pipeline

camera_calibrator = CameraCalibrator()

for current_file in os.listdir("../test_images/"):
    if current_file.__contains__(".jpg"):
        current_image = mpimg.imread("../test_images/" + current_file)
        processed_image = Pipeline(camera_calibrator=camera_calibrator).pipeline(current_image)
        mpimg.imsave("../test_images_output/" + current_file, processed_image)

pipeline = Pipeline(camera_calibrator=camera_calibrator)

white_output = '../output.mp4'
clip1 = VideoFileClip("../project_video.mp4")  # .subclip(0,5)
white_clip = clip1.fl_image(pipeline.pipeline)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
