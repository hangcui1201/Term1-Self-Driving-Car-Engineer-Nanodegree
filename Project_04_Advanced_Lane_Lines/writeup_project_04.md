# Advanced Lane Finding Project  
---

The goals / steps of this project are the following:

(1) Camera calibration using 9 x 6 chessboard images. (camera matrix & distortion coefficients)  
(2) Convert original imagel to undistorted image using camera matrix and distortion coeffiencts.  
(3) Apply perspective transformation (M) to get bird view warped image. (input: undistort image, output: warped image)  
(4) Create binary image using various combinations of color and gradient thresholds on warped images. (Sobel Operator)  
(5) Find lanes on the binary images. (identify lane line pixels)  
(6) Implement sliding windows and fit a polynomial.  
(7) Measure lane curvature.  
(8) Lane prediction and drawing.  

[//]: # (Image References)

[image1]: ./output_images/original_chessboard.jpg "Original Chessboard" 
[image2]: ./output_images/undistort_chessboard.jpg "Undistort Chessboard"
[image3]: ./output_images/test2.jpg "Test2"
[image4]: ./output_images/test_binary.jpg "Test Binary"
[image5]: ./output_images/window_lane_detection.png "Window Lane"
[image6]: ./output_images/lane_prediction.jpg "Lane Prediction"

[video1]: ./project_video_output.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the Jupyter notebook located in "Advanced_Lane_Lines.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Chessboard Image

![alt text][image1]

Undistorted Chessboard Image

![alt text][image2]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 2).  Here's an example of my output for this step.  


![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `undistort_birdview()`, which appears in cell 3. The `undistort_birdview()` function takes as inputs an image (`img`) and camera parameters, i.e. mtx and dist.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[590, 450],
     [260, 680],
     [1045, 680],
     [690, 450]])
dst = np.float32(
    [[300, 0],
     [300, 720],
     [1000, 720],
     [1000, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 300, 0        | 
| 260, 680      | 300, 720      |
| 1045, 680     | 1000, 720     |
| 690, 450      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]  


#### 4. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 7.

#### 5. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 9 in the function `pipline_prediction()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used r channel in RGB format and also s channel, l channel in HLS format for lane detection. R channel is very useful to detect yellow and while pixel. I used Sobel in the x direction to detect the lane and gradient in x direction. Combine the r channel, s channel and the gradient method, basically this approach can give me reasonable result. 

I hand tuned the threshold of s_thresh, r_thresh and sx_thresh. To get more robust result, considering other color channel could be a good option. Or channel different combinations of thresholds to get robust result.  
