import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from HeadFile import training_data_eig_lu_eig_class,training_data_eig_lu_eig_class_all_snr,MDL_MMSE,training_data_eig_class,MDL_subfun_random_angle


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.models import load_model

from keras.optimizers import RMSprop
# from sklearn import preprocessing
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.utils import np_utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

### =================== System Parameters ================


number_antenna = 10   # Antenna number
total_samples = 100000
test_times=10

Hidden_units = [32,16,10]
N_LAYERS = len(Hidden_units)
n_input = number_antenna
n_output_reg =  1
learning_rate = 0.001
mean_error_train = []
batch_size = 256
epoch = 800
n_output_cla=number_antenna

'''varying paprameters'''
# SNRdB_i=5
snap_i=100          # 计算协方差期望所需要的重复次数
SNR_range=range(0,41,5)
# Snap_range=[int(np.round(np.power(10,snap) )) for snap in np.arange(1.2,4.8,0.2)]


#
'''eig_reg ########################################################'''
model_eig_reg = keras.Sequential()
model_eig_reg.add(Dense(Hidden_units[0],activation='selu',input_shape=((n_input,))))
for l_n in range(N_LAYERS-1):
	model_eig_reg.add(Dense(Hidden_units[l_n+1],activation='selu'))
model_eig_reg.add(Dense(n_output_reg))
# model_eig_reg.add(Dense(n_output_reg,activation='selu',input_shape=((n_input,))))

'''lu_reg ########################################################'''
model_lu_reg = keras.Sequential()
model_lu_reg.add(Dense(Hidden_units[0],activation='selu',input_shape=((n_input,))))
for l_n in range(N_LAYERS-1):
	model_lu_reg.add(Dense(Hidden_units[l_n+1],activation='selu'))
model_lu_reg.add(Dense(n_output_reg))

'''eig_class ########################################################'''

model_eig_cla = keras.Sequential()
model_eig_cla.add(Dense(Hidden_units[0],activation='selu',input_shape=((n_input,))))
for l_n in range(N_LAYERS-1):
	model_eig_cla.add(Dense(Hidden_units[l_n+1],activation='selu',kernel_initializer='lecun_normal',bias_initializer=keras.initializers.Constant(value=0.001)))
model_eig_cla.add(Dense(n_output_cla,activation='softmax'))

'''lu_class ########################################################'''

model_lu_cla = keras.Sequential()
model_lu_cla.add(Dense(Hidden_units[0],activation='selu',input_shape=((n_input,))))
for l_n in range(N_LAYERS-1):
	model_lu_cla.add(Dense(Hidden_units[l_n+1],activation='selu',kernel_initializer='lecun_normal',bias_initializer=keras.initializers.Constant(value=0.001)))
model_lu_cla.add(Dense(n_output_cla,activation='softmax'))

### =================== model training ================
InputEig, InputLu, OutputReg, OutputClass=training_data_eig_lu_eig_class_all_snr(total_samples,number_antenna,snap_i)
#
# '''Model training'''
# model_eig_reg.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
# model_eig_reg.fit(InputEig, OutputReg, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2)
#
# model_eig_cla.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
# model_eig_cla.fit(InputEig, OutputClass, batch_size=batch_size, epochs=epoch,verbose=1, validation_split=0.2)

# model_lu_reg.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
# history=model_lu_reg.fit(InputLu, OutputReg, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2)
#
# model_lu_cla.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
# model_lu_cla.fit(InputLu, OutputClass, batch_size=batch_size, epochs=epoch,verbose=1, validation_split=0.2)

print('Model is saving...')
save_path_eig_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_reg_vs_snr_' + str(
	snap_i) + '_ant_' + str(number_antenna)+'_sample_'+str(total_samples) + '_.h5'

save_path_lu_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_lu_reg_vs_snr_' + str(
	snap_i) + '_ant_' + str(number_antenna) +'_sample_'+str(total_samples) + '_.h5'

save_path_eig_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_class_vs_snr_' + str(
	snap_i) + '_ant_' + str(number_antenna)+'_sample_'+str(total_samples)  + '_.h5'

save_path_lu_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_lu_class_vs_snr_' + str(
	snap_i) + '_ant_' + str(number_antenna)+'_sample_'+str(total_samples)  + '_.h5'

# model_lu_cla.save(save_path_lu_cla)
# model_eig_cla.save(save_path_eig_cla)
# model_lu_reg.save(save_path_lu_reg)
# model_eig_reg.save(save_path_eig_reg)


model_eig_cla=load_model(save_path_eig_cla)
model_eig_reg=load_model(save_path_eig_reg)

acc_eig_reg_vs_snap=[]
acc_eig_cla_vs_snap=[]
acc_lu_reg_vs_snap=[]
acc_lu_cla_vs_snap=[]

for SNRdB_i in SNR_range:
	'''Model testing'''

	accuracy_eig_reg=0
	accuracy_eig_class=0
	accuracy_lu_reg=0
	accuracy_lu_class=0
	for test_t in range(test_times):
		print('test: ','SNRdB_i',SNRdB_i,'snap_i',snap_i,'test_t',test_t)
		InputEig_test, InputLu_test, OutputReg_test, OutputClass_test=training_data_eig_lu_eig_class(SNRdB_i,total_samples,number_antenna,snap_i)

		y_pred_eig_reg = model_eig_reg.predict(InputEig_test)
		accuracy_eig_reg += sum((np.around(y_pred_eig_reg)) == OutputReg_test) /(total_samples*test_times)

		y_pred_eig_cla = model_eig_cla.predict(InputEig_test)
		accuracy_eig_class += np.sum(np.argmax(y_pred_eig_cla, axis=1) == np.argmax(OutputClass_test, axis=1)) /(total_samples*test_times)

		# y_pred_lu_reg= model_lu_reg.predict(InputLu_test)
		# accuracy_lu_reg += sum((np.around(y_pred_lu_reg)) == OutputReg_test) / (total_samples * test_times)
		#
		# y_pred_lu_cla = model_lu_cla.predict(InputLu_test)
		# accuracy_lu_class += np.sum(np.argmax(y_pred_lu_cla, axis=1) == np.argmax(OutputClass_test, axis=1)) /(total_samples*test_times)
	acc_eig_reg_vs_snap.append(accuracy_eig_reg)
	acc_eig_cla_vs_snap.append(accuracy_eig_class)
	acc_lu_reg_vs_snap.append(accuracy_lu_reg)
	acc_lu_cla_vs_snap.append(accuracy_lu_class)


'''snr point'''
total_samples = 10000
batch_size = 128
epoch = 800

'''eig_reg ########################################################'''
model_eig_reg = keras.Sequential()
model_eig_reg.add(Dense(Hidden_units[0], activation='selu', input_shape=((n_input,))))
for l_n in range(N_LAYERS - 1):
	model_eig_reg.add(Dense(Hidden_units[l_n + 1], activation='selu'))
model_eig_reg.add(Dense(n_output_reg))
# model_eig_reg.add(Dense(n_output_reg,activation='selu',input_shape=((n_input,))))
'''eig_class ########################################################'''

model_eig_cla = keras.Sequential()
model_eig_cla.add(Dense(Hidden_units[0], activation='selu', input_shape=((n_input,))))
for l_n in range(N_LAYERS - 1):
	model_eig_cla.add(Dense(Hidden_units[l_n + 1], activation='selu', kernel_initializer='lecun_normal',
							bias_initializer=keras.initializers.Constant(value=0.001)))
model_eig_cla.add(Dense(n_output_cla, activation='softmax'))

acc_eig_reg_vs_snr_point=[]
acc_eig_cla_vs_snr_point=[]
for SNRdB_i in SNR_range:

	InputEig, InputLu_test, OutputReg, OutputClass = training_data_eig_lu_eig_class(SNRdB_i,
																						   total_samples,
																						   number_antenna,
																						   snap_i)
	'''Model training'''
	model_eig_reg.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
	model_eig_reg.fit(InputEig, OutputReg, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2)

	model_eig_cla.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
	model_eig_cla.fit(InputEig, OutputClass, batch_size=batch_size, epochs=epoch,verbose=1, validation_split=0.2)

	save_path_eig_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_reg_vs_snr_' + str(SNRdB_i)+'_snap_'+str(
		snap_i) + '_ant_' + str(number_antenna) + '_sample_' + str(total_samples) + '_.h5'

	save_path_eig_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_class_vs_snr_' + str(SNRdB_i)+'_snap_' + str(
		snap_i) + '_ant_' + str(number_antenna) + '_sample_' + str(total_samples) + '_.h5'
	model_eig_cla.save(save_path_eig_cla)
	model_eig_reg.save(save_path_eig_reg)

	'''Model testing'''
	accuracy_eig_reg=0
	accuracy_eig_class=0
	accuracy_lu_reg=0
	accuracy_lu_class=0
	for test_t in range(test_times):
		print('test: ','SNRdB_i',SNRdB_i,'snap_i',snap_i,'test_t',test_t)
		InputEig_test, InputLu_test, OutputReg_test, OutputClass_test=training_data_eig_lu_eig_class(SNRdB_i,total_samples,number_antenna,snap_i)

		y_pred_eig_reg = model_eig_reg.predict(InputEig_test)
		accuracy_eig_reg += sum((np.around(y_pred_eig_reg)) == OutputReg_test) /(total_samples*test_times)

		y_pred_eig_cla = model_eig_cla.predict(InputEig_test)
		accuracy_eig_class += np.sum(np.argmax(y_pred_eig_cla, axis=1) == np.argmax(OutputClass_test, axis=1)) /(total_samples*test_times)

		# y_pred_lu_reg= model_lu_reg.predict(InputLu_test)
		# accuracy_lu_reg += sum((np.around(y_pred_lu_reg)) == OutputReg_test) / (total_samples * test_times)
		#
		# y_pred_lu_cla = model_lu_cla.predict(InputLu_test)
		# accuracy_lu_class += np.sum(np.argmax(y_pred_lu_cla, axis=1) == np.argmax(OutputClass_test, axis=1)) /(total_samples*test_times)
	acc_eig_reg_vs_snr_point.append(accuracy_eig_reg)
	acc_eig_cla_vs_snr_point.append(accuracy_eig_class)


'''MDL '''

acc_MDL=[]
acc_MMSE=[]
acc_AIC=[]
for SNRdB_i in SNR_range:
	print('snap_i',SNRdB_i)
	acc,acc_mmse,acc_aic=MDL_MMSE(SNRdB_i, number_antenna, snap_i)
	acc_MDL.append(acc)
	acc_MMSE.append(acc_mmse)
	acc_AIC.append(acc_aic)

print(acc_MDL)
print(acc_AIC)



Acc_eig_reg_vs_snap=np.asarray(acc_eig_reg_vs_snap)
print(acc_eig_reg_vs_snap)
print(Acc_eig_reg_vs_snap)
Acc_eig_reg_vs_snap=np.reshape(Acc_eig_reg_vs_snap,newshape=(1,np.size(Acc_eig_reg_vs_snap)))
print('eig reg',Acc_eig_reg_vs_snap)
Acc_eig_cla_vs_snap=np.asarray(acc_eig_cla_vs_snap)
Acc_eig_cla_vs_snap= np.reshape(Acc_eig_cla_vs_snap,newshape=(1,np.size(Acc_eig_cla_vs_snap)))
print('eig class',Acc_eig_cla_vs_snap)

Acc_eig_reg_vs_snr_point=np.asarray(acc_eig_reg_vs_snr_point)
Acc_eig_reg_vs_snr_point=np.reshape(Acc_eig_reg_vs_snr_point,newshape=(1,np.size(Acc_eig_reg_vs_snr_point)))
print('eig reg point', Acc_eig_reg_vs_snr_point)
Acc_eig_cla_vs_snr_point=np.asarray(acc_eig_cla_vs_snr_point)
Acc_eig_cla_vs_snr_point= np.reshape(Acc_eig_cla_vs_snr_point,newshape=(1,np.size(Acc_eig_cla_vs_snr_point)))
print('eig class point',Acc_eig_cla_vs_snr_point)

'''
[0.8660599999986933, 0.9165299999984636, 0.9484199999983185, 0.9681399999982288, 0.9805599999981722, 0.9878899999981389, 0.9928399999981163, 0.9952099999981056, 0.9970599999980971]
[0.8697199999986767, 0.918639999998454, 0.9497599999983124, 0.9690699999982245, 0.9810999999981698, 0.988299999998137, 0.9930099999981156, 0.9952999999981051, 0.9971099999980969]
eig reg [[0.995079 0.997433 0.997823 0.998045 0.998039 0.998024 0.998072 0.998131
  0.998096]]
eig class [[0.994953 0.997494 0.997825 0.998061 0.997962 0.998014 0.997979 0.998079
  0.998049]]
eig reg point [[0.99553 0.99724 0.99728 0.99769 0.99812 0.9985  0.99809 0.99801 0.99785]]
eig class point [[0.99514 0.99608 0.99655 0.99776 0.99751 0.99829 0.99682 0.99797 0.99768]]

'''
dataNew = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\Sum_vs_snr_10_5.mat'
# scio.savemat(dataNew, {'acc_MMSE':acc_MMSE,'acc_MDL':acc_MDL,'Acc_lu_cla_vs_snap':Acc_lu_cla_vs_snap,'Acc_lu_reg_vs_snap':Acc_lu_reg_vs_snap,'Acc_eig_cla_vs_snap':Acc_eig_cla_vs_snap,'Acc_eig_reg_vs_snap':Acc_eig_reg_vs_snap})
scio.savemat(dataNew, {'Acc_eig_reg_vs_snr_point':Acc_eig_reg_vs_snr_point,'Acc_eig_cla_vs_snr_point':Acc_eig_cla_vs_snr_point,'acc_AIC':acc_AIC,'acc_MDL':acc_MDL,'Acc_eig_cla_vs_snap':Acc_eig_cla_vs_snap,'Acc_eig_reg_vs_snap':Acc_eig_reg_vs_snap})


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,}


legend_dnn=['m--s','y--x','go-','rv-','k^-','c-*']
g_label=0
# plt.plot(SNR_range ,acc_eig_reg_vs_snr_point,legend_dnn[g_label],markersize=7, label='ERNet, trained separately')
# g_label+=1
# plt.plot(SNR_range ,acc_eig_cla_vs_snr_point,legend_dnn[g_label],markersize=7, label='ECNet, trained separately')
# g_label+=1
plt.plot(SNR_range ,acc_eig_reg_vs_snap,legend_dnn[g_label],markersize=7, label='ERNet')
g_label+=1
plt.plot(SNR_range,acc_eig_cla_vs_snap,legend_dnn[g_label],markersize=7, label='ECNet')
g_label+=1
plt.plot(SNR_range,acc_MDL,legend_dnn[g_label],markersize=7, label='AIC')
g_label+=1
plt.plot(SNR_range,acc_MDL,legend_dnn[g_label],markersize=7, label='MDL')

plt.legend(prop=font1)
plt.ylabel('Accuracy',font1)
plt.xlabel('SNR (dB)',font1)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle="--")
plt.savefig("Pic_DL_vs_snr.png")
plt.savefig('Pic_DL_vs_snr.eps',format='eps')
plt.show()

'''
[0.8640899999987023, 0.9181199999984564, 0.947649999998322, 0.9683299999982279, 0.9798899999981753, 0.9876199999981401, 0.9928299999981164, 0.995119999998106, 0.997319999998096]
[0.8677999999986854, 0.9203899999984461, 0.948979999998316, 0.9692699999982236, 0.980379999998173, 0.9878999999981388, 0.9930799999981152, 0.9952099999981056, 0.9973599999980958]


eig reg [[0.995166 0.997462 0.997819 0.99805  0.998092 0.998087 0.998223 0.998247
  0.998165]]
eig class [[0.995076 0.99757  0.99784  0.997966 0.99798  0.998002 0.998149 0.998157
  0.998051]]
eig reg point [[0.99543 0.99733 0.99753 0.99811 0.99883 0.99836 0.99865 0.99885 0.99859]]
eig class point [[0.99361 0.99698 0.9975  0.99744 0.99823 0.99805 0.99832 0.99817 0.99792]]

'''