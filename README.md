# Solution Design:

The solution to this project was acheived by manually iterating across multiple variables as well as experimenting with the model architecture. Some of the variables adjusted throughout the project included:

Preprocessing:
1. The region of interest: Some of the vertical field of view was masked from the camera images. This was done to remove the car hood from the image and to attempt to stop the model from training on background scenery. Using background scenery when training seems like bad practice and may just result in having the model memorize the driving path. I spend a good deal of time working with the maked data but eventually returned to the full camera view which offered better performance.

2.	The color-space used. Initially normalized RGB data was used in the model. YUV and grayscale were also attempted. RGB data was ultimately used.

3.  Image size. Downsampling the image amounts to throwing away information but it offers increased compute time performance and lower memory requirements. Most importantly, I suspect it scales the image features to a size detectable by the kernels size (thus number of parameters) I could fit into memory.

Model design
1. The general hyper-parameters of the model
	a) batch size: This was initially set at 128 images. It was varied as higher memory requirement models were run, but returned to 128.
	b) number training epochs. While evolving the model and training data, between 5 and 10 training epochs were used. Students noted no increased performance above 5 training epochs. Interestingly, training beyond 6-8 Epochs often resulted in poor performance. 6 epochs were used for initial training and 5 epochs were used for updating data

2. Training data: discussed in a later section
3. Model Architecture: The following factors and sourced influenced the model archtecture
	a) The paper by Bokarski et al. (NVIDIA) reference in the course material titled End to End Learning for Self-Driving Cars. This paper outlines a 5 convultion layer and 4 fully-connector layer network that was successfully used on a Drive-PX PC on a similar problem. This was the initialy network architecture implemented.
	b) The comma.ai github repository giving code for a similar steering angle problem, (perhaps the same problem?). https://github.com/commaai/research/blob/master/train_steering_model.py.
	This model uses a 3 convolution layer, single fully-connected layer network. It used ELU activation which was adopted for my solution after reviewing FAST AND ACCURATE DEEP NETWORK LEARNING EXPONENTIAL LINEAR UNITS (ELU S ) by Clevert et al. This was eventually switched back to Relus.  After reviewing the comma.ai solution, 1 convolution layer was eliminated from the Nvidia style architecture for 4 total convolution layers. One FC layer was eliminated.
	c) Reports from colleagues on Slack and the forums. Other students reported success with fewer layers and lower resolution images. This also influenced the decrease in convolution layers from 5 to 3 as other students were having success with 2 layers.
	d) Intermittently experimenting with adjusting Dropout and Maxpooling between layers.
	The final model architecture consisted of 4 convolution layers adn 3 fully connected layers with a 1 unit output layer. 
4. Model parameters:
	 -Kernel sizes: Initial kernel sizes similar to the Nvidia paper were used of 5x5 or 3x3. Students reported good results with low resoltion image sizes, this lead to attempting to use larger, 32x32 or greater kernel sizes. Many of the features in the data, such as lane markings appeared to on this 30+pixel scale. This didn't work well and finally Kernel sizes of 8x8, 5x5 and 3x3 were used on a downsampled image. I suspect that matching the convolution kernel size to the feature scale of interest (lane markings, etc)was the key to succuss in this assignment.
	 -subsampling sizes: These were used mostly to manage memeory limitations, assuming that less subsampling was desirable, but more subsampling resulted in smaller tensor sizes and lower memory usage. 2X2 subsampling was used on the first 2 convolution layers
	 -padding type for the convolution layers: these were changed between 'valid' and 'same', mostly with consideration to output layer size, with valid padding giving smaller layers. 
	 -Dropout percentages: These were intially accidentally set at 80% which is probably too high. The dropout percentage on layers was vaired between 10 and 30%. 
5. Memory limitations: training was performed on a GTX970 with 4GB of RAM. The GPU RAM seemed to be the limiting factor on how many parameters could be used in the model. The linear layers had to be decreased below the baseline 512 neurons and subsampling had to be introduced in order to meet the GPU memory limitations
6. Normalization: normalization between -1 and 1 was implemented as per the comma.ai github as a lambda function in the Keras model. 

#Model Architecture

The output from Keras with some modifications provides a suitable description of the model used. Also see the output of the Keras visualization: in the attached folder model.png 

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (normalize data -1...1) (None, 64, 128, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (6x6, same pad)  (None, 32, 64, 24)    2616        lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Relu)		         (None, 32, 64, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (10% Dropout)          (None, 32, 64, 24)    0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (5x5 same pad)   (None, 16, 32, 42)    25242       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Relu)     	     (None, 16, 32, 42)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_2 (10% Dropout)          (None, 16, 32, 42)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (4x4 valid pad)  (None, 13, 29, 54)    36342       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_3 (Relu)		         (None, 13, 29, 54)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_3 (20% Dropout)          (None, 13, 29, 54)    0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (3x3 vlida pad)  (None, 11, 27, 64)    31168       dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_4 (Relu)  	         (None, 11, 27, 64)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_4 (20% Dropout)          (None, 11, 27, 64)    0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 19008)         0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_1 (512 FC layer)           (None, 512)           9732608     flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_5 (Relu)		         (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_5 (30% dropout)          (None, 512)           0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_2 (128 FC layer)           (None, 128)           65664       dropout_5[0][0]                  
____________________________________________________________________________________________________
activation_6 (Relu)		         (None, 128)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_6 (10% Dropout)          (None, 128)           0           activation_6[0][0]               
____________________________________________________________________________________________________
dense_3 (128 FC layer)           (None, 12)            1548        dropout_6[0][0]                  
____________________________________________________________________________________________________
activation_7 (Relu)  		     (None, 12)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
output (1 FC layer)              (None, 1)             13          activation_7[0][0]               
====================================================================================================
Total params: 9895201
____________________________________________________________________________________________________





#Training data generation

14 iterations of training data were used while working towards a solution. The Udacity training dataset was perhaps the most important part of acheiving successfull autonomous simulator driving. Based on reports from other students and accumulating experience with this project, successful training datasets consisted of 1-2 laps on the first course (course A) followed by an equal amount of recovery driving data. The data was then copied, mirrored along the horizontal axis and the steering angles were multiplied by -1. This was appended to the original data set as it seemed like a free, easy way to generate additional training data for the inverse scenarios. Most of the track A turned to the left, which may have biased the model to turn to the left without this inverted data.

Recovery driving data consisted of stopping recording, maneuvering the car to the side of the road, or at an approach angle to the side of the road and then starting recording and steering it back into the center of the road. Some recovery data was recorded where the car was aimed at the wall before recording and then steered away.  I found that introducing too much aggresive recovery data into the model would result in wobbly driving, with higher potential to get the vehicle into trouble. 

In some of the training data was generated on the second track (B) and used in the training.

Using the 50Hz Simulator, good driving on most of course A was acheived with reasonable effotrt. The model would fail where lane markigns of the right side of the road would dissapear and the car would drive off course. In attempt to cheat this, I generated training data where the car drove on the left side of the road (Australia!) to follow the consistent lane markings. This worked until it failed. Much more work was required to successfully complete the entire course. 

Using the iverted data, approximately 40000 to 60000 training images were used per track. 

After getting into the ballpark of a solution. I tried generating and retraining on additional data in the specific area of the course where the model was having trouble. This involved running between 1-5 epochs of updating on data generated while driving around a particularly difficult corner. This appeared to not work well and would decrease performance of the initial model. Perhaps retraining on only a small area introduces more complex effects if the optimizer than just letting the model see additional data in an area of interest. 

The notes appended to this report show the training cycles used. Please see the images in example_training_data/ for examples of the downsampled training images. 


#Other considerations

1.Live training: Some students reported implementing live training capability where the user could take over from the model and generate additional data where the model was failing. This seems like a really, really good and convenient idea vs. stopping the autonomous process on failure and retraining, or tuning the data. in retrospect, I regret not developing this earlier on, not knowing the time it would take to successfully implement a working model

2. Tuning an existing model: After training on the Udacity data set, I experimented with generating and retraining on additional data with P3_update_model.py. The final solution involved training on the Udacity data set and then updating on 2 of my data sets. I tried updating on many more data sets (#5, 6, 8, 10, 12, 13, 14) but some of these may have had aggresive recovery driving data which gave me a poor model.

3. Using pre-trained networks. This seemed like a feasible approach but was generally not reported as being required by other students. The other networks were trained on different image data. Their feature recognition training may not have been particularly relevant to this driving simulator. This would have been interesting to test. 

4. Generators: I have 16GB of RAM and another 10GB of swap space. I didn't need to implement a Python generator in this project to train large data sets. They could all be loaded into the memomry. The larger 800MB training set did begin to use up swap space. 
 
5. Testing: The model was tested by letting it run on the simulator. The degree of success was qualitatively measured by how long I could walk away from the computer before returning to find a car in the lake.

6. A proportional velocity controller was implemented to maintain constant velocity.

Conclusion

This assignment involved 2 very significant unknows: What model would be successful for the requiement to drive a simulated car and what training data would be required to train it. Confounding these 2 unknowns along with their sub parameters (preprocessing data, model parameters such as kernel size) resulted in a significant amount of iteration and getting a qualitative feel for the problem space. The availability of the Udacity training data narrowed the problem space by providing some certainty of what type of training should be adequate. Taking notes proved to be valuable. 


Submission: The model submitted appears to drive successfully continously around Track A. The simulator was set to 1152x864 with 'good' resolution





---TRAINING NOTES----
These personal notes may not be completely understandable, but some trends and numbers should be apparent

TrainX = 900MB student training data
TrainY = Udacity training Data Track A
Train # = Robbie iteration number - see notes in file on specific number



3Conv
3FC
Relu
downscale
gray
Train X
----Bad-----


3Conv
3FC
Relu
downscale
gray
Train Y
----Bad-----


5Conv
5FC
Relu
RGB
Train Y
----Bad-----

EXAMPLE: 5C4L = 5 convolution layer, 4 Linear layer

Nvi
TrainY, 10 - fails on sandy corner
TrainY, 10,11 - fails on first corner

NVI 5C4L  5B
TrainX - > fail const angle

NVI 5C2L  5B
TrainX -> fail, car drives to side

NVI 5C2L  5B
TrainX -> fail constant angle?

NVI 2C2L  5B
TrainX -> fail constant angle?

C124L124	5B
TrainY -> OK, train
TrainY,Train12 -> hard bias left
TrainY,Train12M


NVI 5C4L
TrainYM-> Not bad, drives through sandy corner
TrainYM, 12M - > sandy corner
TrainYM, 12M 12M-> fail bridge

Nvi 5C4L, C1: 12x12-64
TrainYM: fail, drives off bridge

Nvi 5C4L, C1: 12x12-112
TrainYM: fail, drives off bridge

------RESTART: fx=0.3, fy=0.3, Cropped 30:150 y axis---------

4C3L L C1 5x5:
TrainYM: probably best so far, end of sand

4C3L C1 12x12x64
TrainYM: wobbles, bad

4C3L C1 7x7x32
TrainYM: drives off sand

4C3L C1 7x7x32
TrainYM: drives off sand

4C3L C1 6x6x24
TrainYM: drives well, hits bridge

4C3L C1 6x6x64
TrainYM: drives well, hits bridge wall??

4C3L C1 6x6x64
TrainYM, 12A: side, sand

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 12A: side, sand

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A,:very good, sand

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A,: good, bridge wall

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A, 5A,: good, wobbles, , gets passed SAND!

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A, 5A,5B,: good,big wobbles, runs off road aft sand

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A, 5A,5B, 6A: good,big wobbles, Unreliable laps!

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A, 5A,5B, 6A,8A: unreliable laps, can drive off the second water edge

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A, 5A,5B, 6A,8A,13A,14A: wobbly, bias left can drive off the second water edge

4C3L C1 6x6x42 C2 5x5x54
TrainYM, 11A, 12A, 5A,5B, 6A,8A,13A,14A,14A: wobbly, can drive off sand


------RESTART: fx=0.4, fy=0.4, full frame---------

MODEL :4C3L C1:6x6x24 C2: 5x5x42 C3: 4x4x54  C4: 3x3x64  L1: 512 L2: 128 L3: 12

Training data used and results

TrainYM: = very good, water2
TrainYM,15A = bad

TrainYM: = very good, water2
TrainYM,14A = doing laps, drives on to edge sometimes
TrainYM,14A,YM,15A,15A = bad
YM, 15A, sand
YM12E - sand

TrainX = not great
4C4L, X =not great

YM,14A,5A = almost perfect, touches edge 1/5 laps - saved 
YM,14A,5A,5B = bad in B, hits wall in  A

YM,14A,5A = almost perfect - saved - SUBMITTED
YM,14A,5A,8A = drives off an edge
YM,14A,5A,8A,12A = not good
