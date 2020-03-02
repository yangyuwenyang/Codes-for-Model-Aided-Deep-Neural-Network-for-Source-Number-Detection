import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from HeadFile import training_data_eig_lu_eig_class,training_data_lu_reg,training_data_eig_reg,training_data_lu_class,MDL_MMSE,training_data_eig_class,MDL_subfun_random_angle,training_data_reg_source_angle_same,training_data_reg_all_snr


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
total_samples = 10000
test_times=10
# Snap_range=[int(np.round(np.power(10,snap) )) for snap in np.arange(1.2,4.7,0.2)]
# Snap_range=[int(np.round(np.power(10,snap) )) for snap in np.arange(1.2,3.,0.2)]
Snap_range=[int(np.round(np.power(10,snap) )) for snap in np.arange(1.0,1.2,0.2)]

Hidden_units = [32,16]
N_LAYERS = len(Hidden_units)
n_input = number_antenna
n_output_reg =  1
learning_rate = 0.001
mean_error_train = []
batch_size = 128
epoch = 400
n_output_cla=number_antenna

'''varying paprameters'''
SNRdB_i= 5
snap_time=100          # 计算协方差期望所需要的重复次数



### =================== model training ================
acc_eig_reg_vs_snap=[]
acc_eig_cla_vs_snap=[]
acc_lu_reg_vs_snap=[]
acc_lu_cla_vs_snap=[]

for snap_i in Snap_range:

	InputEig, InputLu, OutputReg, OutputClass=training_data_eig_lu_eig_class(SNRdB_i,total_samples,number_antenna,snap_i)
	'''eig_class ########################################################'''

	model_eig_cla = keras.Sequential()
	model_eig_cla.add(Dense(Hidden_units[0], activation='selu', input_shape=((n_input,))))
	for l_n in range(N_LAYERS - 1):
		model_eig_cla.add(Dense(Hidden_units[l_n + 1], activation='selu', kernel_initializer='lecun_normal',
								bias_initializer=keras.initializers.Constant(value=0.001)))
	model_eig_cla.add(Dense(n_output_cla, activation='softmax'))
	'''eig_reg ########################################################'''
	model_eig_reg = keras.Sequential()
	model_eig_reg.add(Dense(Hidden_units[0], activation='selu', input_shape=((n_input,))))
	for l_n in range(N_LAYERS - 1):
		model_eig_reg.add(Dense(Hidden_units[l_n + 1], activation='selu'))
	model_eig_reg.add(Dense(n_output_reg))
	# model_eig_reg.add(Dense(n_output_reg,activation='selu',input_shape=((n_input,))))

	'''Model training'''
	model_eig_reg.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
	model_eig_reg.fit(InputEig, OutputReg, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2)

	model_eig_cla.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
	model_eig_cla.fit(InputEig, OutputClass, batch_size=batch_size, epochs=epoch,verbose=1, validation_split=0.2)

	# model_lu_reg.compile(loss='MSE', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
	# history=model_lu_reg.fit(InputLu, OutputReg, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2)
	#
	# model_lu_cla.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
	# model_lu_cla.fit(InputLu, OutputClass, batch_size=batch_size, epochs=epoch,verbose=1, validation_split=0.2)
	print('Model is loading...')
	save_path_eig_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_reg_vs_snap_' + str(
		snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'

	save_path_lu_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_lu_reg_vs_snap_' + str(
		snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'

	save_path_eig_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_class_vs_snap_' + str(
		snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'

	save_path_lu_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_lu_class_vs_snap_' + str(
		snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'

	# # model_lu_cla=load_model(save_path_lu_cla)
	# model_eig_cla=load_model(save_path_eig_cla)
	# # model_lu_reg=load_model(save_path_lu_reg)
	# model_eig_reg=load_model(save_path_eig_reg)

	# model_lu_cla=load_model(save_path_lu_cla)
	model_eig_cla.save(save_path_eig_cla)
	# model_lu_reg=load_model(save_path_lu_reg)
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
	acc_eig_reg_vs_snap.append(accuracy_eig_reg)
	acc_eig_cla_vs_snap.append(accuracy_eig_class)
	acc_lu_reg_vs_snap.append(accuracy_lu_reg)
	acc_lu_cla_vs_snap.append(accuracy_lu_class)

	# print('Model is saving...')
	# save_path_eig_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_reg_vs_snap_' + str(
	# 	snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'
	#
	# save_path_lu_reg = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_lu_reg_vs_snap_' + str(
	# 	snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'
	#
	# save_path_eig_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_eig_class_vs_snap_' + str(
	# 	snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'
	#
	# save_path_lu_cla = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\dl_lu_class_vs_snap_' + str(
	# 	snap_i) + '_snr_' + str(SNRdB_i) + '_ant_' + str(number_antenna) + '_.h5'
	#
	# model_lu_cla.save(save_path_lu_cla)
	# model_eig_cla.save(save_path_eig_cla)
	# model_lu_reg.save(save_path_lu_reg)
	# model_eig_reg.save(save_path_eig_reg)

'''MDL '''

acc_MDL=[]
acc_MMSE=[]
acc_AIC=[]
for snap_i in Snap_range:
	print('snap_i',snap_i)
	acc,acc_mmse,acc_aic=MDL_MMSE(SNRdB_i, number_antenna, snap_i)
	acc_MDL.append(acc)
	acc_MMSE.append(acc_mmse)
	acc_AIC.append(acc_aic)





# Acc_eig_reg_vs_snap=np.asarray(acc_eig_reg_vs_snap)
# print(acc_eig_reg_vs_snap)
# print(Acc_eig_reg_vs_snap)
# Acc_eig_reg_vs_snap=np.reshape(Acc_eig_reg_vs_snap,newshape=(1,np.size(Acc_eig_reg_vs_snap)))
# print('eig reg',Acc_eig_reg_vs_snap)
# Acc_eig_cla_vs_snap=np.asarray(acc_eig_cla_vs_snap)
# Acc_eig_cla_vs_snap= np.reshape(Acc_eig_cla_vs_snap,newshape=(1,np.size(Acc_eig_cla_vs_snap)))
# print('eig class',Acc_eig_cla_vs_snap)
# # Acc_lu_reg_vs_snap=np.asarray(acc_lu_reg_vs_snap)
# # Acc_lu_reg_vs_snap=np.reshape(Acc_lu_reg_vs_snap,newshape=(1,np.size(Acc_lu_reg_vs_snap)))
# # print('lu reg', Acc_lu_reg_vs_snap)
# # Acc_lu_cla_vs_snap=np.asarray(acc_lu_cla_vs_snap)
# # Acc_lu_cla_vs_snap= np.reshape(Acc_lu_cla_vs_snap,newshape=(1,np.size(Acc_lu_cla_vs_snap)))
# # print('lu class',Acc_lu_cla_vs_snap)

print(acc_eig_reg_vs_snap)
print(acc_eig_cla_vs_snap)
print(acc_MDL)
print(acc_AIC)
# [0.9843999999999079]
# acc_MDL=[0.9546999999999112, 0.9611999999999105] 4.6,4.8
# acc_MDL=[0.8701999999999205, 0.9098999999999161, 0.919799999999915, 0.9311999999999138, 0.9315999999999137,0.9326999999999136]
# acc_MMSE=[0.8266999999999253, 0.8854999999999188, 0.919499999999915, 0.9423999999999125, 0.9568999999999109,0.9660999999999099]


# acc_eig_reg_vs_snap= [0.94118 ,0.99116, 0.99992, 1. ,     1.  ,    1.     ]
# acc_eig_cla_vs_snap= [0.94272, 0.99088 ,0.99992 ,1.  ,    1. ,     1.     ]
# acc_lu_reg_vs_snap =[0.89604, 0.98306, 0.9998,  1.   ,   1.     , 1.     ]
# acc_lu_cla_vs_snap= [0.89504, 0.98498 ,0.9997  ,1.   ,   1.    ,  1.     ]

# dataNew = 'E:\CODE_SOURECE_NUMBER_DECTION\CODE_2\ModelSave\Sum_vs_snap.mat'
# scio.savemat(dataNew, {'acc_AIC':acc_AIC,'acc_MDL':acc_MDL,'Acc_eig_cla_vs_snap':Acc_eig_cla_vs_snap,'Acc_eig_reg_vs_snap':Acc_eig_reg_vs_snap})


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,}
acc_cov=[0.37906, 0.33272, 0.48749999999999993, 0.49752, 0.4988600000000001, 0.50308, 0.51386, 0.55614, 0.6114, 0.64606, 0.6569400000000001]
legend_dnn=['bx-','go-','rv-','k^-','c-*','b--s']
g_label=0
plt.semilogx(Snap_range ,acc_eig_reg_vs_snap,legend_dnn[g_label],markersize=7, label='ERNet')
g_label+=1
plt.semilogx(Snap_range,acc_eig_cla_vs_snap,legend_dnn[g_label],markersize=7, label='ECNet')
g_label+=1
# # plt.semilogx(Snap_range ,acc_lu_reg_vs_snap,legend_dnn[g_label],markersize=7, label='LRNet')
# # g_label+=1
plt.semilogx(Snap_range,acc_AIC,legend_dnn[g_label],markersize=7, label='AIC')
g_label+=1
plt.semilogx(Snap_range,acc_MDL,legend_dnn[g_label],markersize=7, label='MDL')
# g_label+=1
# plt.semilogx(Snap_range,acc_cov,legend_dnn[g_label],markersize=7, label='Covariance matrix based network')

plt.legend(prop=font1)
plt.ylabel('Accuracy',font1)
plt.xlabel('Number of snapshots',font1)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle="--")
plt.savefig('Pic_DL_vs_snap_ab.png')
plt.savefig('Pic_DL_vs_snap_ab.eps',format='eps')
plt.show()
# plt.legend(['ERNet','ECNet','LRNet','LCNet','MDL','MMSE MDL'])



# plt.semilogx(Snap_range ,acc_eig_reg_vs_snap)
# plt.semilogx(Snap_range,acc_eig_cla_vs_snap)
# plt.semilogx(Snap_range ,acc_lu_reg_vs_snap)
# plt.semilogx(Snap_range,acc_lu_cla_vs_snap)
# plt.semilogx(Snap_range,acc_MDL)
# plt.semilogx(Snap_range,acc_MMSE)

# plt.semilogx(Snap_range,acc_AIC)

# # plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Number of snapshots')
# # # # plt.legend(['eig_reg','eig_reg_testing','MDL'])
# # plt.legend(['ERNet','ECNet','MDL','MMSE MDL'])
# # # # plt.legend(['eig_reg', 'eig_cla','lu_reg','lu_cla'])
# plt.legend(['ERNet','ECNet','LRNet','LCNet','MDL','MMSE MDL'])
# plt.savefig('temp.png')
# plt.savefig('temp.eps', format='eps')
# plt.show()


