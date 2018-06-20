# **Traffic Sign Recognition** 


### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)

    I put the dataset under the data folder, which is a subfolder of the 
    project folder.

* Explore, summarize and visualize the data set


* Design, train and test a model architecture


* Use the model to make predictions on new images


* Analyze the softmax probabilities of the new images


* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/Training_Set_Bar.png "Training_Set_Bar"
[image2]: ./pictures/Validation_Set_Bar.png "Validation_Set_Bar"
[image3]: ./pictures/Testing_Set_Bar.png "Testing_Set_Bar"

[image4]: ./pictures/25.jpg "Traffic Sign 1"
[image5]: ./pictures/8.jpg "Traffic Sign 2"
[image6]: ./pictures/9.jpg "Traffic Sign 3"
[image7]: ./pictures/14.jpg "Traffic Sign 4"
[image8]: ./pictures/3.jpg "Traffic Sign 5"
[image9]: ./pictures/19.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of the training, validation and testing set.

![alt text][image1]

![alt text][image2]

![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because due to the paper grayscale images for training will get better performance and grayscale images are easy to deal with. 

I also normalized the image data so that the data will have mean zero and equal variance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (LeNet5):


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Flatten				|												|
| Fully connected		| input 400, output 120        					|
| Fully connected		| input 120, output 84        					|
| RELU					|												|
| Dropout				|												|
| Softmax				|            									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model with hyperparameters' setting as followings:

epochs = 300

batch_size = 128

learning_rate = 0.001

optimizer: AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 95.9%

* test set accuracy of 94%

If a well known architecture was chosen:

* What architecture was chosen?

      I choose LeNet5 as the base architecture of CNN.

* Why did you believe it would be relevant to the traffic sign application?

      LeNet5 was initially used for recognizing letters on images. Based on this architecture it is pontentially suitable for small scale image recognition. 

* How does the final model's accuracy on the validation and test set provide evidence that the model is working well?

	  I forget to print the training accuracy. However, based on the validation accuracy (95.9) and testing accuaracy (94%) and corresponding hyperparameters's settings. I think this model works as intended. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

![alt text][image5] 

![alt text][image6] 

![alt text][image7] 

![alt text][image8]

![alt text][image9]

The first image might be difficult to classify because the background of the image could potentially produces big noise for image recognition.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			              |     Prediction	        					| 
|:---------------------------:|:-------------------------------------------:| 
| Road work      		      | Road work  									| 
| Speed limit (120km/h)       | Speed limit (30km/h)					    |
| No passing			      | No passing									|
| Stop	      		          | Stop					 				    |
| Speed limit (60km/h)	      | Speed limit (60km/h)      					|
| Dangerous curve to the left | Dangerous curve to the left   			    |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 94%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a Road Work sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work  									| 
| 0     				| Dangerous curve to the right					|
| 0	     				| Keep right									|
| 0	        			| Ahead only					 				|
| 0 				    | Beware of ice/snow     						|


For the second image, the model is sure that this is a Speed limit (30km/h) sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)  						| 
| 0     				| Speed limit (60km/h)				         	|
| 0	     				| Bicycles crossing								|
| 0	        			| Speed limit (80km/h)					 		|
| 0 				    | Speed limit (20km/h)    						|


For the third image, the model is sure that this is a No passing sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No passing  						            | 
| 0     				| Speed limit (20km/h)				         	|
| 0	     				| Speed limit (30km/h)							|
| 0	        			| Speed limit (50km/h)					 		|
| 0 				    | Speed limit (60km/h)     						|

For the fourth image, the model is sure that this is a Stop sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop  					                	| 
| 0     				| No entry				                    	|
| 0	     				| Turn right ahead								|
| 0	        			| Yield					 	                	|
| 0 				    | Turn left ahead    			     			|

For the fifth image, the model is sure that this is a Speed limit (60km/h) sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (60km/h)  						| 
| 0     				| Speed limit (80km/h)				         	|
| 0	     				| Bicycles crossing								|
| 0	        			| Wild animals crossing				 		    |
| 0 				    | Speed limit (50km/h)    						|

For the sixth image, the model is sure that this is a Dangerous curve to the left sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Dangerous curve to the left  	     			| 
| 0     				| Slippery road				                 	|
| 0	     				| Speed limit (20km/h)				         	|
| 0	        			| Speed limit (30km/h)				         	|
| 0 				    | Speed limit (50km/h)    						|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


