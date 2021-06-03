#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[7]:


net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
# load classes from coco 
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    #classes = f.read().splitlines()

layer_names = net.getLayerNames()
output_layer = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,(len(classes),3))

cap = cv2.VideoCapture(0)

#frame_width = int(cap.get(3))
#frame_height = int(cap.get(3))

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        img = frame
        height,width,depth = img.shape
        
        blob = cv2.dnn.blobFromImage(img,0.003,(416,416),(0,0,0),True,crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer)
        
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for det in out:
                scores = det[5:] # starting from 6th element to the end
                # scores --> array used to store all the 80 classes prediction
                class_id = np.argmax(scores) 
                confidence = scores[class_id] 
                if confidence > 0.6:
                    cx = int(det[0]*width)
                    cy= int(det[1]*height)

                    w = int(det[2]*width)
                    h = int(det[3]*height)

                    #x = int(cx -w/2)
                    #y = int(cx -h/2)

                    x,y = int(cx-w/2), int(cy-h/2)
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        n_det = len(boxes)
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
        for i in range(n_det):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = classes[class_ids[i]]
                c = str(confidences[i])
                cv2.rectangle(img,(x,y),(x+h,y+w),(255,0,0),3)
                cv2.putText(img,f'{label} {c[0:4]}',(x,y+50),cv2.FONT_HERSHEY_PLAIN,5,(0,255,0),5)
                
                
        cv2.imshow('Yolo-Live',img)
        if cv2.waitKey(25) & 0xFF ==ord('q'):
            break
    else:
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




