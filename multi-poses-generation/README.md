# Multiple Head Poses Generation

This folder contains the main code and experiments for `multiple head poses generation`.

## Installing tips

1 - Download the configs from here: https://drive.google.com/drive/folders/1dzwQNZNMppFVShLYoLEfU3EOj3tCeXOD?usp=sharing

2 - place the files inside **train.config** folder and **3ddfa/train.config** folder

3 - Download the 3d fitting model from here: 

https://drive.google.com/file/d/18UQfDkGNzotKoFV0Lh_O-HnXsp1ABdjl/view

4 - Place the downloaded .tar file without extracting it in the current folder.

5 - clone the neural renderer repo and install it using **pip install .** (Make sure to have torch==1.12 installed)

6 - the **FaceRenderer** class exists in **change_pose.py**


## Running the code

1 - Make an object from FaceRenderer class

``` renderer = FaceRenderer() ```

2 - Call the **rotate_face** function by feeding for it the image as numpy array (Make sure that the image is in BGR format) and the target angle.

```rotated_face = renderer.rotate_face(img, angle=30)```

3 - In order to reuse the last generated 3d model which is saved already on the desk, You can pass **reuse** argument as follows

```rotated_face = renderer.rotate_face(img, angle=30, reuse=True)```


4 - Never hesitate to contact Abo El Rmamez.