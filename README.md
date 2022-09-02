# Minor-Project-2019-VEHICLE-IDENTIFICATION-USING-DEEP-LEARNING-MODEL
A minor project on training a deep learning model to identify vehicle


Table of Contents
1. Introduction
1.1. Scope of the Work
1.2. Product Scenarios
2. Requirement Analysis
2.1. Functional Requirements
2.2. Non-functional Requirements
2.3. Use Case Scenarios
2.4. Other Software Engineering Methodologies (as applicable)
3. System Design
3.1. Design Goals
3.2. System Architecture
3.3. Detailed Design Methodologies (as applicable)
4. Work Done
4.1. Development Environment.
4.2. as required.
4.3. Results and Discussion
4.4. Individual Contribution of project members
5. Conclusion and Future
Work 5.1. Proposed Workplan of the project
References Appendix (optional)

List of Figures List of Tables
1. Training Dataset
2. Testing Dataset
3. Model Architecture Using Summary()
4. Results
5. Image Having Name 1.jpg
6. Prediction 1
7. Image Having Name 2.jpg
8. Prediction 2
9. Image Having Name 3.jpg
10. Prediction 3

1. Introduction
1.1. Scope of the Work
This Vehicle Recognition Systems use the concept of optical recognition to recognize
the type of vehicle in the image. In other words, it takes the image of a vehicle as the
input and outputs the type of vehicle in the image. This system uses the image to
detect whether there is vehicle in it and then find out which type of vehicle it is from
existing trained model.
Due to less complex structure of this system as it uses images to identify cars and not
characters on license plate this system can be easily used, modified and new vehicles
can be added to the model and model retrained with new classes.
1.2. Product Scenarios
In today&#39;s populated world there is growing no of people having vehicles but the
government cannot keep up with so many vehicles by hiring more manpower so the
traffic system should be mostly automated to keep up with this many vehicles.
One of the problem is vehicle identification for traffic violation, parking, traffic
management etc.
Scenario 1
A car comes to park in parking lot there is not enough space for four wheelers but
only two wheelers so only two wheelers will be allowed. In an automated car parking
system one of system has to be to recognize the type of vehicle entering the parking
lot..
Scenario 2
This product can be used to recognize a car type which is allowed on a specific road
like trucks not allowed on small roads. One specific Indian scenario is that trucks

coming outside from cities in India are not allowed to go inside the city at peak hours
of traffic.
Scenario 3
This product can also be used to manage traffic by diverting heavy vehicles like bus
and trucks to less busy routes when there is peak traffic thus reducing more traffic in
already congested roads.

2. Requirement Analysis
2.4. Functional Requirements
Only a low powered PC or laptop to load weights and model architecture for
predicting new images but will need a centralized high power PC to retrain model if
new class is to be added.
2.5. Non-functional Requirements
Image Should clearly show the vehicle
The dataset should have all types of vehicles to train the algorithm.
The dataset should be stored in in database which can be easily accessed by the algorithm.

2.6. Use Case Scenarios
In parking system a PC can be installed to be used to classify vehicles by loading
Tensorflow and Python on PC and using JSON and h5 file to load the pre-trained
model. Let a car come to parking then image is taken of car and fed to the model
which classifies where the vehicle has to be parked according to their size. This
reduces the need of parking attendant who tells us where to park.
Another example is traffic management to stop heavy vehicles like truck and bus from
making already congested roads more congested. This can be done by a parallel
application which will divert heavy vehicles from busy roads by identifying them in
images taken from traffic camera.
2.4. Other Software Engineering Methodologies (as applicable)
There are many pre-trained model available but they require very complex coding for
custom datatset to be made by user. Like Xception,VGG16,VGG19,ResNet,
ResNetV2, ResNeXt, InceptionV3, InceptionResNetV2 ,MobileNet and
MobileNetV2.

3. System Design
3.4. Design Goals
To predict and classify type of vehicles from 6 classes available.
In this project we are using these methods to make our model: -
I). The model should differentiate between
a) Cars
b) Two wheelers
c) Trucks,
d) Bus
e) Background
f) Non-Motorized Vehicle

![image2](https://user-images.githubusercontent.com/49300118/188106170-e45cd874-e6d9-41fd-b211-921aa1c459f9.png)
FIG 1: TRAINING DATASET 1
![image3](https://user-images.githubusercontent.com/49300118/188106171-63de3657-88a7-43e1-a462-928e84339851.png)
FIG 2: TESTING DATASET 1

3.5. System Architecture
![image4](https://user-images.githubusercontent.com/49300118/188106173-799b82b1-1096-4576-9473-168987fa4474.png)
FIG 3: MODEL ARCHITECTURE USING SUMMARY()
4. Work Done 

            4.1.Development Environment. 
Spyder in Anaconda GUI using Tensorflow backend with Keras.



as required. 
Gathering Information about the existing image processing systems using different methods like machine learning, Deep Neural Network, OpenCV, etc.
Doing Literature Survey for machine learning and Deep Learning.
Doing course I machine learning and deep learning through FORSK Technology
Using one of the methods found above to make our algorithm.This will be made in Anoconda’s Python IDE will use some existing libraries.
Collecting dataset for training of algorithm -This step is used for collection of data against which images are to be compared.
We used MIO-TCD Dataset as base dataset which we changed to suit our model needs
Writing the code for algorithm in Python IDE. 
Training the algorithm using collected data. In this step we compare the collected dataset against the existing dataset and know accuracy of our algorithm.
Testing this algorithm in real world application using images of vehicles on road. 






























Results and Discussion 




FIG 4:RESULTS









![image5](https://user-images.githubusercontent.com/49300118/188106175-8a999542-60e4-4e7d-806f-1cae551d7626.png)
![image6](https://user-images.githubusercontent.com/49300118/188106176-426b9c6e-7539-484a-91d7-37bd920c1b6b.jpg)
![image7](https://user-images.githubusercontent.com/49300118/188106180-b77a14b8-5fbb-4f5b-ad0c-b6b6f232bc7f.png)
![image8](https://user-images.githubusercontent.com/49300118/188106181-89581a4a-5872-4e42-be96-51bcf0ff0d10.jpg)
![image9](https://user-images.githubusercontent.com/49300118/188106183-b96146a3-63df-467b-9ad4-c5bd9cc427b6.png)
![image10](https://user-images.githubusercontent.com/49300118/188106186-88573b58-3269-41bd-a423-dce3b8407a25.jpg)
![image11](https://user-images.githubusercontent.com/49300118/188106188-116b6354-cab9-4c30-883e-4c9751e6b720.png)














PREDICTIONS 
1.

                     FIG 5 :IMAGE HAVING NAME 1.JPG
            		FIG 6: PREDICTION 1

2


















































	


2.
 

                     FIG 7 :IMAGE HAVING NAME 3.JPG



FIG 8:PREDICTION 2














3.

                     FIG 9 :IMAGE HAVING NAME 2.JPG





FIG 10:PREDICTION 3

 



Individual Contribution of project members 
I am doing this project alone.



5. Conclusion and Future 

Work 5.1. 
Using this trained model in conjunction with front end application like parking system cameras, traffic cameras to classify vehicles and help in parking and traffic management. We can further extend this model to include more detailed classes like 
in cars we can further classify them as sedan, hatchback, SUV, etc. Another example is further classification of two wheelers into motorcycle, bicycle, scooter, etc.
We can extend this model to use localization of vehicles in images. 


References Appendix 
Machine Learning – Forsk OpenEdx Course
 Nakano N, Yasui N, inventors; Panasonic Corp, assignee. Vehicle recognition apparatus. United States patent US 5,487,116. 1996 Jan 23.
 Kato T, Ninomiya Y, Masaki I. Preceding vehicle recognition based on learning from sample images. IEEE Transactions on Intelligent Transportation Systems. 2002 Dec;3(4):252-60.
Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
 4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013  (3dRR-13). Sydney, Australia. Dec. 8, 2013.
  Borman WM, Walker DL, inventors; Motorola Solutions Inc, assignee. Automatic vehicle monitoring identification location alarm and voice communications system. United States patent US 3,644,883. 1972 Feb 22.
Z. Luo, F.B.Charron, C.Lemaire, J.Konrad, S.Li, A.Mishra, A. Achkar, J. Eichel, P-M Jodoin 
MIO-TCD: A new benchmark dataset for vehicle classification and localization
in press at IEEE Transactions on Image Processing, 2018
