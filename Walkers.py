import cv2


# Create our body classifier
body_cascade = cv2.CascadeClassifier('c:/Users/livcr/OneDrive/Pictures/PRO-106-ProjectTemplate-main/106/haarcascade_frontalface_default.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('c:Users/livcr/OneDrive/Pictures/projects/PRO-106-ProjectTemplate-main/106/walking.avi')

# Loop once video is successfully loaded
while (True):
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    peple = body_cascade.detectMultiScale(gray,1.2,3)
    print(peple)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in peple:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame',frame)
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
