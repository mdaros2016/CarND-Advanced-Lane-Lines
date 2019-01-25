## Writeup Michele Da Ros


--- 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[video1]: ./output.mp4 "Video"

[calibration]: ./camera_cal/calibration1.jpg
[calibrated]: ./output_images/camera_cal_calibration1_calibrated.jpg

[distorted]: ./test_images/straight_lines1.jpg
[undistorted]: ./output_images/straight_lines1_undistorted.jpg

[edges]: ./output_images/edges.jpg

[lines_for_perspective_transform]: ./output_images/undistorted_image_with_source_points_drawn.jpg
[bird_eye]: ./output_images/bird_eye_on_straight_line.jpg

[no_artificial_points]: ./output_images/test7_no_artificial_points.jpg
[artificial_points]: ./output_images/test7.jpg

[sliding_windows]: ./output_images/sliding_windows.png
[detect_lines_from_prior]: ./output_images/detect_lines_from_prior.png

[final_result]: ./output_images/test6.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration and distortion correction are implemented in the class [CameraCalibrator](./src/cameraCalibrator.py).
It's implemented exactly as shown in the relative lesson. 
For calibrating the camera, all the images of the folder [camera_cal](./camera_cal) are parsed, 
and the corners detected with the function `cv2.findChessboardCorners` are added to a list.
For every successful corner detection, the set of real world images are added to another list 
(all the chessboards used for calibration have the same dimension, so the same points are added at every iteration)
The transformation matrix is computed with the function `cv2.calibrateCamera`

The file [cameraCalibratorTest](./src/cameraCalibratorTest.py) helps manually testing the camera calibrator, since it initializes the coefficients of CameraCalibrator and undistorts a test image


![before calibration][calibration]*before calibration* 
![calibrated][calibrated]*calibrated*

 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion correction is implemented in the method `undistort` of the class [CameraCalibrator](./src/cameraCalibrator.py).

The transformation matrix is initialized lazily with the first invocation of `undistort` (using a set of calibration pictures), and stored in the object properties
The output of [cameraCalibratorTest](./src/cameraCalibratorTest.py) described above shows the effect of distortion correction on a test image

![before distortion correction][distorted]*before distortion correction* 
![after distortion correction][undistorted]*after distortion correction*




#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

[EdgesDetector](./src/edgesDetector.py)`#detectEdges` is used for creating thresholded binary image, where only the pixels that may be part on a lane line are white.

It checks that the horizontal gradient of the luminosity, or the saturation meet the threshold.

Additionally, since we can predict where the lane lines will be, only the pixels the that fit in a trapezoid with the base on the bottom of the picture and the short side at 40% height are kept
 
The file [edgesDetectorTest](./src/edgesDetectorTest.py) helps manually testing `EdgesDetector`, since it detects the edges in a test image, and saves the resulting thresholded binary image
 
![ thresholded binary image][edges]
*thresholded binary image*

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is implemented in the class [PerspectiveTransformer](./src/perspectiveTransformer.py)

This class can be initialized with any set of source and destination points, but a set of points for this project has been provided as default parameters of the constructor

The perspective transformations between camera view and bird-eye view are performed by the methods `toBirdEye` and `toOriginal`

The file [PerspectiveTransformerTest](./src/perspectiveTransformerTest.py) helps manually testing `Perspective Transformer`, since shows an image with the source points used for the transformation, and the same image transformed to bird-eye view.


![alt text][lines_for_perspective_transform]
*source points*

![alt text][bird_eye]
*bird eye view*

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for detecting the pixels of the lane, and fitting them with a polynomial is implemented in the file [LinesDetector](./src/linesDetector.py)

The function   `fitPolinomial` takes a binary image tha represent a bird-eye view of the street, 
and the 2 objects that describe the 2 lane lines (left and right) that have been recently calculated in the last iterations,
and returns the lane lines(left and right) that have been detected, and an image that can be visualized for debugging purposes

Like suggested in the project description, the detected lines are instances of the class [Line](./src/line.py).

`fitPolinomial` works like that:
* If no lines were detected in the past iteration, detect the pixels that compose the lane lines using the sliding windows technique (through the function `find_lane_pixels`)
* Otherwise, detect the lane pixels using a slightly modified implementation of the technique described in the lesson "Finding the lines: search from prior" (through the function `find_lane_pixels_from_prior`)
* For each detected line, fit the corresponding pixels with a 2nd order polynomial from y to x using `np.polyfit`,
* For each detected line, store the result of the detection in every correspondent `Line object`, through the method `Line#update_fitted`, that works like that 
    * Push the latest detected polynomial coefficients at the top of a list that keeps track of the last polynomials (`line#recent_fit`)
    * Compute the polynomial (from y to x) for every y point of the image (from 0 to 720), and store the x values at the top of a list that keeps track of the last plotted x values (`line#recent_plotx`)
    * If necessary, trim the lists `line#recent_fit` and `line#recent_plotx`, to delete old values that will not considered any more when computing the average for detecting the current lines
    * For every y point, compute the average of the value of fitted polynomial over the memorized values of `line#recent_plotx`, and save the result to `line#best_plotx`
    * `line#best_plotx` is the result that will be shown
    

`find_lane_pixels ` is implemented pretty much like seen in the lesson "Finding the Lines: Sliding Window", but with a hacky customization.
I noticed that in some occasions only the portions of the line more close to the driver were included in the thresholded binary image.
For this reason, only some points close to the bottom of the image were used for fitting a polynomial: this led to  curves that were diverging from the real lane line on the top of the picture.
To fix this problem, I decided to add an "artificial" pixel (used for fitting the polynomial) at the center of each window that contains less than 50 pixels.
The effect of this hack is shown below


![without arficial points][no_artificial_points]
*without artificial points*

![with arficial points][artificial_points]
*with artificial points*



`find_lane_pixels_from_prior` uses the same principles described in the corresponding lesson, 
but instead of using the polynomial coefficients of the previous fitted line as reference for searching pixels in the current frame, 
the average coordinates of the last fitted lines (stored in `line#best_plotx`) are used
  

The pictures below show how sliding windows and search from prior detect the lane pixels

![alt text][sliding_windows]
*sdetect lines with sliding window*

![alt text][detect_lines_from_prior]
*detect lines from prior*


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for detecting the curvature radius and the offset is implemented in the file [curvatureDetector](./src/curvatureDetector.py)

The curvature is computed by the function `measure_curvature` like this:

* Roughly compute the meter/pixel ratio on the bird-eye image (considering that on one bird-eye frame fit 9 dotted lines, 
that have more or less the same length as a 4-5m long car ) 
* Transform the detected bird-eye lines coordinates from pixels to meters, and fit a polynomial to the points in this space
* Calculate the average curvature radius of the left line over the last 25 frames (in meters), using the formula seen in the lecture
* Calculate the average curvature radius of the right line over the last 25 frames
* Return the average between the curvature of the left and of the right line

The offset from the center is computed by the function `measure_offset_real`, like this:
* Calculate the value of x coordinate of the left line and the right line at the bottom of the screen (y=720)
* Calculate the center of the lane as the average x value of the 2 lines computed at the step above
* Calulate the distance in pixel between the center of the image and the center of the lane (through a simple substraction)
* Convert the distance in meters by multiply the value in pixel by the scaling factor (xm_per_pix)


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The original picture is annotated with the lane and the information about curvature and offset via  [pictureAnnotator](./src/pictureAnnotator.py)`#decorate`

The lane lines are drawn like this:
* Produce a black "mask" image, of the same dimension of the "bird eye view" image used for lane detection
* In this mask picture, fill the space between the left and the right line with green
* Transform the mask picture from bird eye view to camera view 
(the same instance of `PerspectiveTransformer` used for producing the "warped" image is injected in an instance of `PictureAnnotator`)
* The transformed mask is overlapped to the initial picture

Information about curvature radius and offset are printed as text over the image by the method `write_offset`


![final result][final_result]
*final result*

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)
---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging aspects that I've faced are corner extraction and line detection.
It's hard to have a good balance between being too strict with edges detection (that leads to having few pixels for 
fitting polynomials, resulting in poor line detection) or being too permissive (that leads to noise)

I've noticed that my pipeline fails on the "challenge" videos,
probably because it gets tricked by features of the image (like the shadow lines) that are interpreted as lane lines,
although they are not.

Maybe, for making the pipeline more robust, we could not try to fit a polynomial to all the colored pixels that fall into
specific positions, but we should try to detect patterns formed by these pixels 
(consider the 2 parallel rims of the lane lines, or the rectangles that form the lines) 

