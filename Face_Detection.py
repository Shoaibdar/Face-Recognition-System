#!/usr/bin/env python
# coding: utf-8

# # Face_Detection

# In[1]:


# HOG Algorithim
import PIL.Image  #  displays image and draw lines on the screen
import PIL.ImageDraw
import face_recognition  #dectets face

#load image
image = face_recognition.load_image_file("Pictures/Pictures/imgcache0.26381100.jpg")



#find all images
#HOG Pre-trained model
face_locations = face_recognition.face_locations(image)
#face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)

number_of_faces = len(face_locations)
print("Number of faces : ", number_of_faces)

pil_image = PIL.Image.fromarray(image)

for face_location in face_locations:
    top, right, bottom, left = face_location
    print("Top, Left, Bottom, Right : ", top, left, bottom, right)
    
    draw = PIL.ImageDraw.Draw(pil_image)
    draw = draw.rectangle([left, top, right, bottom], outline = "red", width = 5)
    
pil_image.show()


# # Face_Landmarks

# In[2]:


import PIL.Image
import PIL.ImageDraw
import face_recognition

#load image
image = face_recognition.load_image_file("Pictures/Pictures/IMG_20180223_141403736.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)

number_of_faces = len(face_landmarks_list)
print("Number of person: ",number_of_faces)

pil_image = PIL.Image.fromarray(image)

draw = PIL.ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    for name, list_of_points in face_landmarks.items():
        
        print(name, list_of_points)
        
        draw.line(list_of_points, fill = "red", width = 3)
        
pil_image.show()


# # Face_Encoding

# In[3]:


import face_recognition

image = face_recognition.load_image_file("Pictures/Pictures/imgcache0.29015754.jpg")

#face encoding
face_encodings = face_recognition.face_encodings(image)

if (len(face_encodings) == 0):
    print("No faces were found")
else:
    first_face_encoding = face_encodings[0]
    print(first_face_encoding) #prints 128
    


# # Face_Recognition System

# In[4]:


import face_recognition

image_of_person_1 = face_recognition.load_image_file("Pictures/Pictures/IMG_20180222_132058817.jpg")
image_of_person_2 = face_recognition.load_image_file("Pictures/Pictures/IMG_20180316_101131038.jpg")
image_of_person_3 = face_recognition.load_image_file("Pictures/Pictures/IMG_20180305_140305782.jpg")

person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0] #Put this ->[0] only if only one person is in picture
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]
person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]

#create list of known faces
known_face_encodings = [
    person_1_face_encoding, 
    person_2_face_encoding, 
    person_3_face_encoding
]

#load image of unknown person
unknown_image = face_recognition.load_image_file("Pictures/Pictures/IMG_20180306_103928557.jpg")

#face encoding
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

for unknown_face_encoding in unknown_face_encodings:
    result = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)#compares the images
    
    #name = "Unknown"
    if result[0]:
        name = "person 1"
    elif result[1]:
        name = "person 2"
    elif result[3]:
        name = "person 3"
    else:
        print("Match not found")
        
    print("Found ", name, " in the photo")


# # Digital_Makeup

# In[5]:


import PIL.Image
import PIL.ImageDraw
import face_recognition

image = face_recognition.load_image_file("Pictures/Pictures/IMG_20180321_170905803.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = PIL.Image.fromarray(image)

d = PIL.ImageDraw.Draw(pil_image, "RGBA")

for face_landmarks in face_landmarks_list:
    
    d.line(face_landmarks["left_eyebrow"], fill = (128, 0, 128, 100), width = 3)
    d.line(face_landmarks["right_eyebrow"],fill = (128, 0, 128, 100), width = 3)
    
    d.polygon(face_landmarks["top_lip"],fill = (128, 0, 128, 100))
    d.polygon(face_landmarks["bottom_lip"],fill = (128, 0, 128, 100))
    
pil_image.show()


# # Face_Recognition System Tuning

# In[6]:


import face_recognition

image_of_person_1 = face_recognition.load_image_file("Past your image1 path")
image_of_person_2 = face_recognition.load_image_file("Past your image2 path")
image_of_person_3 = face_recognition.load_image_file("Past your image3 path")

person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0] #Put this ->[0] only if only one person is in picture
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]
person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]

#create list of known faces
known_face_encodings = [
    person_1_face_encoding, 
    person_2_face_encoding, 
    person_3_face_encoding
]

#load image of unknown person
unknown_image = face_recognition.load_image_file("Pictures/Pictures/imgcache0.6874806.jpg")

#face encoding
face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2) #default upsample is 1, 2 means it will zoom twice the pic
unknown_face_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=face_locations)

for unknown_face_encoding in unknown_face_encodings:
    result = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)#compares the images
    
    #name = "Unknown"
    if result[0]:
        name = "person 1"
    elif result[1]:
        name = "person 2"
    elif result[3]:
        name = "person 3"
    else:
        print("Match not found")
        
    print("Found ", name, " in the photo")


# In[ ]:


image_of_person_1 = face_recognition.load_image_file("Pictures/Pictures/IMG_20180222_132058817.jpg")
image_of_person_2 = face_recognition.load_image_file("Pictures/Pictures/IMG_20180316_101131038.jpg")
image_of_person_3 = face_recognition.load_image_file("Pictures/Pictures/IMG_20180305_140305782.jpg")

