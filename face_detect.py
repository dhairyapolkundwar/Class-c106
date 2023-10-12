import cv2

img = cv2.imread("boy.jpg")

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray)
print(len(faces))

for (x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
       # Crop Image To Save Image
       cropped = img[y:y+h, x:x+w]
       cv2.imwrite("face.jpg", cropped)
             
cv2.imshow('img',img)
cv2.waitKey(0)



