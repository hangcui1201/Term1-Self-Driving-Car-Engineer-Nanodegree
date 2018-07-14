# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/center.jpg "Center"
[image2]: ./pictures/left.jpg "Left"
[image3]: ./pictures/right.jpg "Right"
[image4]: ./pictures/center_crop.jpg "Center_Crop"
[image5]: ./pictures/left_crop.jpg "Left_Crop"
[image6]: ./pictures/right_crop.jpg "Right_Crop"
[image7]: ./pictures/center_flip.jpg "Center_Flip"
[image8]: ./pictures/left_flip.jpg "Left_Flip"
[image9]: ./pictures/right_flip.jpg "Right_Flip"
[image10]: ./pictures/steering_angle.jpg "Steering Angle"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup\_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a Nvidia-like convolution neural network defined in model.py lines 87-122. It has 5 convolutional layers, for which the first three convolutional layers have 5x5 filter sizes and last two have 3x3 filter sizes. The first three convolutional layers stride with (2,2), depths between 24 and 64. After the convolutional layer, I use Flatten() functions and 3 fully-connected layers, also using relu as the activation function. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 90). I also crop the input image using Cropping2D function to get the useful portion of the images, which remove the top 70 pixels and 25 pixels form the bottom. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with dropout equals to 0.5 in order to reduce overfitting (model.py lines 118 and 120). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 124). I trained the model with center, left, and right images, and have a correction steering angle parameter for "correcting" the steering angles for left and right images (model.py line 21). 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make use of convolution neural network and I use the Nvidia net as a start. I thought this model might be appropriate because it has been successfully implemented in the real self-driving car for Nvidia test. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set but a low mean squared error on the validation set. This implied that the model was underfitting. 

To combat the underfitting, I collect more data for training, I collected 4 counter clockwise laps and 2 clockwise laps around 15611 x 3 images.

Then I did data augmentation, namely flip the images and steering angles horizontally (model.py line 51-64). I also use left and right images for training the model and use a hyperparameter correction for tuning the steering angles for the left and right images (model.py line 21).

The final step was to run the simulator to see how well the car was driving around track one. I collected the data with average speed around 20 mph. So I tested my trained model at 20 mph. The vehicle is able to drive autonomously around the track without leaving the road. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 87-122) consisted of a convolution neural network with the following layers and layer sizes:

After cropping the image the input shape is 65 x 320 x 3

Convolution layer 1: 65 x 320 x 3 -> 31 x 158 x 24

Convolution layer 2: 31 x 158 x 24 -> 14 x 77 x 36

Convolution layer 3: 14 x 77 x 36  -> 5 x 37 x 48

Convolution layer 4: 5 x 37 x 48 -> 3 x 35 x 64

Convolution layer 5: 3 x 35 x 64 -> 1 x 33 x 64

Fully-connected layer 1: 100

Fully-connected layer 1: 50

Fully-connected layer 1: 10

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

Image from center camera

![alt text][image1]

Image from left camera

![alt text][image2]

Image from right camera

![alt text][image3]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to center itself. Then I repeated this process on track one with the opposite direction of the previous collection in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize the data better. For example, here is an image that has then been flipped:

Flipped Image from center camera

![alt text][image7]

Flipped Image from left camera

![alt text][image8]

Flipped Image from right camera

![alt text][image9]


Before feeding the images into the model, I cropped the images in order to get the useful portion for training. I remove the 70 pixels from the top and 25 pixels from the bottom. Here are the examples:

Cropped Image from center camera

![alt text][image4]

Cropped Image from left camera

![alt text][image5]

Cropped Image from right camera

![alt text][image6]


After the collection process, I had 15611 number of data points for training the model. I finally randomly shuffled the data set and put 20% of the data into a validation set. Here is the distribution of the steering angles in the collected data set. 

Steering Angle Distribution

![alt text][image10]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by observing the loss values on the training and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
