import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from myutils import MultiImageFlow

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

count, s = 100, list(range(10,100+10,10))

flow = None

def test(mode = 1):
	global flow, get

	try:
		del flow, get
	except:
		pass
	np.random.shuffle(s)

	if mode == 1:
		print('Test All Unit')
		x_train = np.random.random((100,s[0],s[1],3))
		y_train = np.random.random((100,s[2],s[3],3))
		flow = MultiImageFlow(datagen,x_train,y_train,32)
		get = flow.__getitem__(0)
		print(*(i.shape for i in get))
		
	if mode == 2:
		print('Test Unit, List')
		x_train = [np.random.random((100,s[0],s[1],3)),np.random.random((100,s[2],s[3],3)),np.random.random((100,s[4],s[5],3))]
		y_train = [np.random.random((100,s[6],s[7],3)),np.random.random((100,s[8],s[9],3))]
		flow = MultiImageFlow(datagen,x_train,y_train,32)
		get = flow.__getitem__(0)
		for i in (0,1):
			print(i)
			print(*(i.shape for i in get[i]),sep=',\n')

	if mode == 3:
		print('Test List, List')
		x_train = [np.random.random((100,s[0],s[1],3)),np.random.random((100,s[2],s[3],3)),np.random.random((100,s[4],s[5],3))]
		y_train = [np.random.random((100,s[6],s[7],3)),np.random.random((100,s[8],s[9],3))]
		flow = MultiImageFlow([datagen]*len(x_train),x_train,y_train,32)
		get = flow.__getitem__(0)
		for i in (0,1):
			print(i)
			print(*(i.shape for i in get[i]),sep=',\n')

	if mode == 4:
		print('Test Unit, Dict')
		x_train = [np.random.random((100,s[0],s[1],3)),np.random.random((100,s[2],s[3],3)),np.random.random((100,s[4],s[5],3))]
		y_train = [np.random.random((100,s[6],s[7],3)),np.random.random((100,s[8],s[9],3))]
		x_train = {chr(i+ord('A')):j for i,j in enumerate(x_train)}
		y_train = {chr(i+ord('G')):j for i,j in enumerate(y_train)}
		flow = MultiImageFlow(datagen,x_train,y_train,32)
		get = flow.__getitem__(0)
		for i in (0,1):
			print(i)
			print(*({key:get[i][key].shape for key in get[i]}.items()),sep=',\n')

	if mode == 5:
		print('Test Dict, Dict')
		x_train = [np.random.random((100,s[0],s[1],3)),np.random.random((100,s[2],s[3],3)),np.random.random((100,s[4],s[5],3))]
		y_train = [np.random.random((100,s[6],s[7],3)),np.random.random((100,s[8],s[9],3))]
		x_train = {chr(i+ord('A')):j for i,j in enumerate(x_train)}
		y_train = {chr(i+ord('G')):j for i,j in enumerate(y_train)}
		flow = MultiImageFlow({i:datagen for i in x_train},x_train,y_train,32)
		get = flow.__getitem__(0)
		for i in (0,1):
			print(i)
			print(*({key:get[i][key].shape for key in get[i]}.items()),sep=',\n')
	


for i in range(5):
	test(i+1)