import cv2 as cv 

#Using Haar Cascade filters for feature detection in images 

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +  'haarcascade_eye_tree_eyeglasses.xml') #Trained to detect eyes in pictures with people wearing glasses 
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

vid=cv.VideoCapture(0)

while True:
    _,frame= vid.read()
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  #Algo works better with grayscale images 
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    eyes=eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)  # Higher scaling and minNeighbors for eyes and smile
    smile=smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,255,255),2)
    for (x,y,w,h) in eyes:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
    for (x,y,w,h) in smile:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
    cv.imshow('Face Detection', frame)
    

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()