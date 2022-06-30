import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Kane Williamson','Virat Kohli']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\Meer Zamanat Ali\Desktop\Face_detect\Faces\Virat Kohli\Virat_2.jpg')
#Virat_2,6,8

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print('Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]),(100,20), cv.FONT_HERSHEY_COMPLEX,1.0, (0,0,255), thickness= 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness= 8)

cv.imshow('Detected Face', img)

cv.waitKey(0)