# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
def recfunc(a):
    # read input image
    image = cv2.imread(a)

    if image is None:
        print("Could not read input image")
        exit()

    # preprocessing
    output = np.copy(image)

    # load pre-trained model
    model = load_model('gender_pridiction.model')

    # face-detection in image
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces=faceCascade.detectMultiScale(output,1.3,5)
    for (x,y,w,h) in faces:
        p=13
        output = cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
        roi_image = image[y-p:y+h+p, x-p:x+w+p]
        roi_image = cv2.resize(roi_image, (90,90))
        roi_image = roi_image.astype("float") / 255.0
        roi_image = img_to_array(roi_image)
        roi_image = np.expand_dims(roi_image, axis=0)

   # run inference on input image
        confidence = model.predict(roi_image)[0]

   # write predicted gender and confidence on image (top-left corner)
        classes = ["M", "F"]    
        idx = np.argmax(confidence)
        label = classes[idx]
        label = "{} {:.2f}%".format(label, confidence[idx] * 100)
        cv2.putText(output, label, (x,y),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)

    # print confidence for each class in terminal
    print(image)
    print(classes)
    print(confidence)
    print(label)    

    # save output image
    cv2.imwrite("output.jpg", output)
    
    # press any key to close image window
    #cv2.waitKey()

    # release resources
    cv2.destroyAllWindows()
