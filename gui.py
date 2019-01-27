from tkinter import *
import cv2, os
from PIL.ImageTk import PhotoImage
from PIL import Image
from keras.models import load_model,model_from_json
import numpy as np


def load_predictor_models():
	json_file = open('model/aae_decoder.json','r').read()
	gen = model_from_json(json_file)
	del json_file
	gen.load_weights('model/aae_decoder_weights.hdf5')
	return gen


root = Tk()
root.title('GANesha')
G = load_predictor_models()
sliders = []
no = 0
z = np.random.normal(size=(1,100))

class SliderClass:
	def __init__(self,master,i,j):
		global no
		self.no = no
		self.w = Scale(master, from_=10, to=-10, tickinterval=0.1)
		self.w.grid(row=i,column=j)
		self.w.bind("<ButtonRelease-1>", self.change)
		self.w.set(np.clip(z[0,self.no]*10,-10.0,10.0))
		no +=1

	def change(self, event):
		z[0,self.no] = self.w.get()/2
		im = G.predict(z)
		im = (0.5 * im + 0.5)*255
		im = im[0,:,:,:].astype('uint8')
		im = cv2.resize(im,(150,150))
		im = Image.fromarray(im)
		im = PhotoImage(im)
		panel.configure(image=im)
		root.mainloop()


im = G.predict(z)
im = (0.5 * im + 0.5)*255
im = im[0,:,:,:].astype('uint8')
im = cv2.resize(im,(150,150))
im = Image.fromarray(im)
im = PhotoImage(im)
panel = Label(root, image = im, width=200,height=200)
panel.grid(rowspan=5,column=0)

r,c = 5,20
for i in range(r):
	for j in range(1,c+1):
		s = SliderClass(root,i,j)
		sliders.append(s)


root.mainloop()