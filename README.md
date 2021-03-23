# Face-Recognition
This project is running as live video stream which will capture a video through webcam and face will be detected.  
And through further processing the images will be converted to an .yml files and on that the recognition task will be done.  

Here the projects description is done :  
**1.face_dataset** file uses computer vision for face detection using **haarcascade** xml file of face, the video is captured and images are stored in folder.  
This file also takes the name of user so that at time of reognition the user is recognised by its name.

**2.face_labels**file uses some pyhon library to perform some file reading the dataset which we got are then labelled here and a trainer.yml file is created which consist of   
labels for users.

**3.face_predict** file recognizes the user with some percantage of being true.Here the Computer vision face recognizer i.e.**LBPHFaceRecognizer** was used to recognize different users    

The **4.face-recognition-sytem** file is comnbined code of all 3 files which perform all the three task in single file  

Here are some pictures you will be seeing for your **dataset** folder  and your detected and recognised face.
