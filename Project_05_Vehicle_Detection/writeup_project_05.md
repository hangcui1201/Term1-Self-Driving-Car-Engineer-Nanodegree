## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_test_1.jpg
[image2]: ./output_images/hog_vehicle_test_1.jpg
[image3]: ./output_images/vehicle_test_2.jpg
[image4]: ./output_images/hog_vehicle_test_2.jpg
[image5]: ./output_images/non_vehicle_test.jpg
[image6]: ./output_images/hog_non_vehicle_test.jpg
[image7]: ./output_images/windows.jpg
[image8]: ./output_images/on_windows.jpg
[image9]: ./output_images/vehicle_heatmap.jpg
[image10]: ./output_images/vehicle_ex.jpg
[image11]: ./output_images/non_vehicle_ex.jpg

[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the Jupyter notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images in second code cell of Jupyter notebook.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image10]
![alt text][image11]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are some example HOG feature extraction images and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]
![alt text][image2]

![alt text][image3]
![alt text][image4]

![alt text][image5]
![alt text][image6]


#### 2. Explain how you settled on your final choice of HOG parameters.

I did function extract_features() in the fourth code cell of Jupyter notebook. I started with the code examples from lesson 23: Combine and Normalize Feature. I did the HOG feature extraction in `YCrCb` color space and tried varies of parameters on orientations, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block using suggested parameters given in the lecture. It turns out the default parameters just work fine for me thought it may not be optimal. It is really a good practice to do the trial and error to find the good combination of the parameters. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I prepared the extracted training features in the fifth code cell of the Jupyter notebook. And I use a linear SVM to train classifier model in the six code cell of the Jupyter notebook. I split up data into randomized training and test sets and let the test_size to be 0.3. Then I generate a scalar using the command StandardScaler().fit(X_train), applying it to both X_train and X_test. The final test accuracy goes up to 99.15%


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I did the sliding window search in eighth code cell of the Jupyter notebook. Initially, I set x_start_stop to [None, None] and y_start_stop to [390, 670]. I tried out [390, 670] of y_start_stop working fine. The I tried the parameter xy_window value. At the beginning, I set two values of xy_window to be the same. It turns out it does not work very well since the vehicles are not necessary to be a square, then I tried different values of xy_window, making it rectangles. After trial and error, I decide to use six windows to do the combination and the parameters of xy_window are shown in the eighth code cell of the Jupyter notebook.  

Here is the example. 


![alt text][image7]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is 

![alt text][image8]
---


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I did a heatmap to average detection results from images in the ninth code cell of the Jupyter notebook. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a test image

![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have two main issues when I was working on this project. 

The first is parameter tuning, we have several parameters to tune such as HOG parameters, overlap ration, search windows number, etc. Each of them could result in different performance of the pipeline, spending time doing parameter tuning is important, also a headache. 

Second, the efficiency of my pipeline or computational time. The approach I used in this project (SVM & HOG) does not perform fast enough for real-time application. Even though it is fine for this project. 

The trained classifier can potentially fails even thought it was trained on approximately 15000 images. And the changes in lighting and shadows of the environment will still cause the model to make false positive detection. Data augmentation could be a good choice to make the model more robust. 