# CCPD



Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline



# CCPD: Chinese City Parking Dataset



This repository is designed to provide an open-source dataset for license plate detection and recognition, described in _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. 



## Specification of the categorise above:



- **sample**: gives 6 example pictures for each sub-dataset(blur/challenge/db/fn/np/rotate/tilt).



- **rpnet**: The training code for a license plate localization network and an end-to-end network which can detect the license plate bounding box and recognize the corresponding license plate number in a single forward.



- **ccpd_base.zip**: contains 1000 pictures which are taken from different perspectives and different distances, under different illuminations and in different. 



- **ccpd_blur.zip**: contains 1000 pictures where pictures are blurred largely.



- **ccpd_challenge.zip**: contains 1000 pictures which is the most difficult benchmark for LPDR algorithm.



- **ccpd_characters.zip**: contains numerical and character images which is designed for training neural networks to recognize segmented character images.



- **ccpd_db.zip**: contains 1000 pictures where illuminations on the LP area are dark or extremely bright. 



- **ccpd_fn.zip**: contains 1000 pictures where the distance from the LP to the shooting location is relatively far or very near.



- **ccpd_np.zip**: contains 1000 pictures where the car in the picture dose not own a LP.



- **ccpd_rotate.zip**: contains 1000 pictures with great horizontal tilt degree.



- **ccpd_tilt.zip**: contains 1000 pictures with both relatively great horizontal tilt degree and vertical tilt degree.



- **ccpd_weather.zip**: contains 1000 pictures which are taken in rainy weather.





## Annotations

Annotations are embedded in file name.

A sample image name is "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg". Each name can be splited into seven fields. Those fields are explained as follows.

- **Area**: Area ratio of license plate area to the entire picture area.

- **Tilt degree**: Horizontal tilt degree and vertical tilt degree.

- **Bounding box coordinates**: The coordinates of the left-up and the right-bottom vertices.

- **Four vertices locations**: The exact (x, y) coordinates of the four vertices of LP in the whole image. These coordinates start from the right-bottom vertex.

- **License plate number**: Each image in CCPD has only one LP. Each LP number is comprised of a Chinese character, a letter, and five letters or numbers.

- **Brightness**: The brightness of the license plate region.

- **Blurriness**: The Blurriness of the license plate region.



## Training instructions

Input parameters are well commented in python codes. You can increase the batchSize as long as enough GPU memory is available.



#### For convinence, we provide a well-trained wR2 model named "wR2.pth221" in the rpnet/ folder for easy training RPnet.



First train the localization network defined in wR2.py as follows:

```

  python wR2.py -i [IMG FOLDERS] -b 4

```

After wR2 finetunes, we train the RPnet defined in rpnet.py. Please specify the variable wR2Path (the path of the well-trained wR2 model) in rpnet.py.

```

  python rpnet.py -i [TRAIN IMG FOLDERS] -b 4 -se 0 -f [MODEL SAVE FOLDER] -t [TEST IMG FOLDERS]

```



## Test demo instructions

After fine-tuning RPnet, you need to uncompress a zip folder and select it as the test directory. The argument after -s is a folder for storing failure cases.

```

  python rpnetEval.py -m [MODEL PATH, like /**/fh02.pth] -i [TEST DIR] -s [FAILURE SAVE DIR]

```



## Acknowledgement

If you have any problems about CCPD, please contact detectrecog@gmail.com.



Please cite the paper _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_, if you benefit from this dataset.
