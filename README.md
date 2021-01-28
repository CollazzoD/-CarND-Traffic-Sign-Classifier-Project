## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./images/signs.png "Distinct Signs in the Dataset"
[image2]: ./images/distribution.png "Sign's Distribution in the Dataset"
[image3]: ./myTrafficSigns/Ahead%20only.jpg "Ahead only"
[image4]: ./myTrafficSigns/Bicycles%20crossing.jpg "Bicycles crossing"
[image5]: ./myTrafficSigns/No%20entry.jpg "No entry"
[image6]: ./myTrafficSigns/Wild%20animals%20crossing.jpg "Wild animals crossing"
[image7]: ./myTrafficSigns/Pedestrians.jpg "Pedestrians"

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Well...you're reading it! And here is a link to my [project code](https://github.com/CollazzoD/-CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The training set contains 34799 images
* The validation set contains 4410 images
* The test set contains 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First, I present you all the distinct signs inside the dataset.

![alt text][image1]

Now, we're going to see how's the distribution of these signs inside the dataset!

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to NOT preprocess the images, just to see what was the result. I was able to achieve a maximum accuracy of 0.901, which was not enough (the rubric points reaquire at least 0.93).

So, as a next step I decided to convert the images to grayscale. This is because several images in the training are pretty dark and contain only little color: the grayscaling helps in achieving a better accuracy.

Furthermore, several studies (like [this one](https://www.sciencedirect.com/science/article/abs/pii/0305048396000102
)) recommend to normalize the data to improve the performance of the neural networks, so I decided to go with a simple normalization using the following formula `(pixel - 128)/ 128`, which converts the int values of each pixel [0,255] to float values with range [-1,1].

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is based on the LeNet model architecture. I added dropout layers before each fully connected layer in order to prevent overfitting. My final model consisted of the following layers:

| Layer                  |     Description                                |
|------------------------|------------------------------------------------|
| Input                  | 32x32x1 gray scale image                       |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 14x14x6                   |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 5x5x16                    |
| Flatten                | outputs 400                                    |
| Dropout            |                                                |
| Fully connected        | outputs 120                                    |
| RELU                   |                                                |
| Dropout            |                                                |
| Fully connected        | outputs 84                                     |
| RELU                   |                                                |
| Dropout            |                                                |
| Fully connected        | outputs 43                                     |
| Softmax                |                                                |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer and the following hyperparameters:
* batch size: 128
* number of epochs: 150
* learning rate: 0.0006
* keep probability of the dropout layer: 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I used an iterative approach for the optimization of validation accuracy:
1. First of all, I took the original LeNet model from the course. I had to modify the model in order to feed in input colored images from the training set with shape (32,32,3). I also modified the number of outputs so that it fitted to the 43 unique labels in the training set. The training accuracy was **0.901** but only one of the five new images was correctly classified! The hyperparameters used where the following: 
    *   EPOCHS = 50 
    *   BATCH_SIZE = 128
    *   learning_rate = 0,001 

2. After adding the grayscaling preprocessing the validation accuracy increased to **0.912** 
   (same hyperparameters as step 1). From now on, all the five new images where correctly classified, so I tried to achieve a better accuracy with some modification.

3. After adding the normalization of the training and validation data the validation accuracy increased to **0.928** (hyperparameters unmodified)

4. Now I tried to change the hyperparameters and achieved a validation accuracy of **0.911**. The results suggested some overfitting and I tried to overcome this in the next step. As for now, the hyperparameters used where:
    *   EPOCHS = 50 
    *   BATCH_SIZE = 128
    *   learning_rate = 0,0007 

5. It was time to add a dropout layer: I added it after RELU of final fully connected layer and obtained a validation accuracy of **0.931** 
    *   EPOCHS = 50 
    *   BATCH_SIZE = 128
    *   learning_rate = 0,0007 
    *   dropout probability = 0.5

6. Since I was not satisfied with the results, I tried to add a dropout before each fully connected layer. I kept the same hyperparameters as before and obtained a validation accuracy of **0.936**

7. After some other experiments (I won't write the all here for the sake of simplicity) I reached a validation accuracy of **** with the following hyperparameters:


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] 
![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7]

The triangular signs might be difficult to classify since they're somehow similar to each other (only the internal image is changing, so the neural network cannot rely only to the triangular shape). The same applies to the circular signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| Bycicles crossing     | Bycicles crossing								|
| No entry				| No entry										|
| Pedestrians      		| Pedestrians   				 				|
| Wild animals crossing	| Wild animals crossing 						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

* You can download the German Traffic Sign Dataset from [this site](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

* This project uses a pickled dataset downloaded from the classroom's link in the "Project Instructions" content. In this dataset the images are already to 32x32, and contains a training, validation and test set.
