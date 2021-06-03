#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2
import numpy as np

# matplot uses --->RGB channel
# cv2 uses     --->BGR channel


# In[37]:


# loading Yolo models(weights and config)
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')


# In[38]:


# load classes from coco 
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    #classes = f.read().splitlines()


# In[39]:


# load the target image that we want to do the object detection
my_img = cv2.imread('D:\image.jpg')
my_img = cv2.resize(my_img,(1280,720))
ht, wt, _ = my_img.shape
# also the convert the BGR image into RGB image i.e. swapRB = True
# input image is in different format
# but the darknet or the yolo takes the image in different format
# --> so you need to convert your image before feeding into the yolo network
blob = cv2.dnn.blobFromImage(my_img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)
blob.shape


# In[40]:


# to feeding this blob into your network, we can write
# since our saved model is inside the 'net' object
net.setInput(blob)


# In[41]:


# Now we want to get the o/p from the network,for that we need to define the last layer of the NN
# this function getUnconnectedOutLayersNames --> is just to get the ouput layers name
last_layer = net.getUnconnectedOutLayersNames()
# This will get you the names of the last layer


# In[42]:


# If you want to get the output from the last layer then
layer_out = net.forward(last_layer)
#layer_out[0].shape
#layer_out[0][0]


# In[43]:


boxes = []
confidences = []
class_ids = []

for output in layer_out:
    for detection in output:
        score = detection[5:] # probability of each class are coming after 5th element
        # to identify that particular element we need to apply numpy argmax method
        # it returns the highest value index from the array
        class_id = np.argmax(score) 
        # class_id has given the index of the element, confidence will give the probability of that element
        confidence = score[class_id]
        if confidence > 0.6:
            center_x = int(detection[0]*wt)
            center_y = int(detection[1]*ht)
            w = int(detection[2]*wt)
            h = int(detection[3]*ht)
# here the detection will be between o and 1 and to convert that into
# normal image size it is multiplied by width and height
            
# Now since we got the centerx,y and your width, height
# we want to figure out the value in the x-dimension and y-dimension
            x = int(center_x -w/2)
            y = int(center_x -h/2)
        
            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)


# In[44]:


# when we do the object detection, more than one bounding boxes happens,
# so we use the non-maximum suppression
# to only keep the highest scores boxes
#print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#indexes
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxes),3)) # gives different colored bounding 
                                                      # for two or more images in same image 


# In[45]:


# for drawing bounding boxes on top of images
for i in indexes.flatten(): # flatten --> we need this in a single list
    x,y,w,h = boxes[i]  # get x,y,h,w coordinates from boxes
    label = str(classes[class_ids[i]]) # in label we want to give the class name 
    confidence = str(round(confidences[i],2)) # round-off to 2 decimal number
    color = colors[i]
    
    cv2.rectangle(my_img,(x,y), (x+w,y+h),color,2)
                #(image, initial coordinates, final coordniates)
    cv2.putText(my_img,label +" "+confidence,(x,y+20),font,2,(0,0,0),2)
    
cv2.imshow('img',my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[36]:


# once we have the image then we need to convert it into input image that can be fit into yolo frame
# first of all we need to resize the image into 
# we need to normalize it by dividing the pixel value by 255
# and the values are also intended to be in RGB order

# since our saved model is inside the 'net' object
#net.setInput(blob) # this setInput is used to set the input from the blob into the network
# we need to define the output layer names 
# and from net we get the layer names
# this function getUnconnectedOutLayersNames --> is just to get the ouput layers name
#output_layers_name = net.getUnconnectedOutLayersNames()
#layerOutputs = net.forward(output_layers_name)

# we need the extract the bounding boxes, confidences and the predicted classes
#into different lists
#first we inilialize the list


# In[ ]:





# In[ ]:





# In[ ]:




