
import numpy as np
from keras.utils import load_img, img_to_array
import os

# import PySimpleGUI as sg
# sg.theme("DarkTeal2")
# layout = [[sg.T("")], [sg.Text("Choose a folder: "), sg.Input(key="-IN2-" ,change_submits=True), sg.FolderBrowse(key="-IN-")],[sg.Button("Submit")]]

# ###Building Window
# window = sg.Window('My File Browser', layout, size=(600,150))
    
# while True:
#     event, values = window.read()
#     print(values["-IN2-"])
#     if event == sg.WIN_CLOSED or event=="Exit":
#         break
#     elif event == "Submit":
#         path = values["-IN-"]
#         img = load_img(path, target_size=(300, 300))
#         x = img_to_array(img)
#         x /= 255
#         x = np.expand_dims(x, axis=0)
#         classes = model.predict(x)
#         print(classes[0])
#         if classes[0]>0.5:
#            print(i + " is a human")
#         else:
#            print(i + " is a horse")

import PySimpleGUI as sg
path= sg.popup_get_file('Enter the file you wish to predict:')
img = load_img(path, target_size=(300, 300))
x = img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
classes = model.predict(x)
print(classes[0])
result =""
if classes[0]>0.5:
    result = "The image seems to be of a human"
    print(" is a human")
else:
    result = "The image seems to be of a horse"
    print(" is a horse")
sg.popup('You image prediction based on the CNN model is:', result)