import cv2
import numpy as np
from PIL import Image
import os

#Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create() #here the trained data will be stored
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#function to get images and associate IDs and labels
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #path array
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') #converted to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1]) #extracting ID from the image name
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids #returning images with their IDs in two arrays

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, IDs = getImagesAndLabels(path)
recognizer.train(faces, np.array(IDs)) #training

# Saving the model into trainer/trainer.yml
recognizer.save('trainer/trainer.yml')

# Printing the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(IDs))))
