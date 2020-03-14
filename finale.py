
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime,date
import pymysql
from keras import backend as K
from flask import Flask,jsonify,request

app = Flask(__name__)



# In[2]:

@app.route("/")
def fn1():

    #face expression recognizer initialization
    from keras.models import model_from_json
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('expp.h5') #load weights


    # In[3]:


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    a=0
    b=0
    c=0



    while(True):
        ret, img = cap.read()

        img = cv2.resize(img, (640, 360))
        img = img[0:308,:]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            if w > 30: #trick: ignore small faces
                cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face

                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)

                img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

                #------------------------------

                predictions = model.predict(img_pixels) #store probabilities of 7 expressions
                max_index = np.argmax(predictions[0])

                #background of expression list
                overlay = img.copy()
                opacity = 0.4
                cv2.rectangle(img,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

                #connect face and expressions
                cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
                cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)

                emotion = ""
                for i in range(len(predictions[0])):
                    emotion = "%s %s%s" % (emotions[i], round(predictions[0][i]*100, 2), '%')
                    print(emotions[max_index])
                    #emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    #print(max_index)
                    if(max_index==3 or max_index==5):
                        a=a+1            
                    elif(max_index==0 or max_index==1 or max_index==2 or max_index==4):
                        b=b+1
                    else:
                        c=c+1
                    
                    

                    """if i != max_index:
                        color = (255,0,0)"""

                    color = (255,255,255)

                    cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                #-------------------------

        #cv2.imshow('img',img)
        #cv2.resizeWindow(img, 1920, 720)
        cv2.imshow('img',img)
        
        
        

        #---------------------------------

        

        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit

            break


    d=180
    a=a//d
    b=b//d
    c=c//d
    print(a,b,c)
    #kill open cv things
    cap.release()
    cv2.destroyAllWindows()
    K.clear_session()

    detection_loginId = request.args.get("loginId")
    print("loginId=",detection_loginId)
    positiveCount = str(a)
    negativeCount = str(b) 
    neutralCount = str(c)
    detectionDate = str(date.today())
    detectionTime = str(datetime.time(datetime.now()))

    connection = pymysql.connect(
            host = "localhost",
            user = "root",
            password = "root",
            db ="visage"
        )
    cursor1 = connection.cursor()
    cursor1.execute(
            "INSERT INTO detection(positiveCount,negativeCount,neutralCount,detectionDate,detectionTime,detection_loginId) VALUES('"+positiveCount+"','"+negativeCount+"','"+neutralCount+"','"+detectionDate+"','"+detectionTime+"','"+str(detection_loginId)+"')"
        )
    connection.commit()
    cursor1.close()
    connection.close()


    infoDict = {"positive":a,"negative":b,"neutral":c}

    response = jsonify(infoDict)

    response.headers.add("Access-Control-Allow-Origin","*")

    return response
    

app.run()
