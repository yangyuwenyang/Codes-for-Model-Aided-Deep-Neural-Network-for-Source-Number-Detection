
from __future__ import division
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import scipy.io as sio
import keras
from keras.layers import   Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
# from sklearn import preprocessing
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.utils import np_utils
from scipy.linalg import cholesky

Mapping = {1: np.array([[1]]), 2: np.array([[0.2, 0.4], [0.4, 1]]), 3: np.array([[0.2, 0.4, 0.1], [0.4, 1, 0.1], [0.1, 0.1, 1]]),
		   4: np.array([[0.2, 0.4, 0.1, 0.2], [0.4, 1, 0.1, 0.3], [0.1, 0.1, 1, 0.3], [0.2, 0.3, 0.3, 1]]),
		   5:np.array([[0.2, 0.4, 0.1, 0.2, 0.1], [0.4, 1, 0.1, 0.3,0.1], [0.1, 0.1, 1, 0.3,0.2], [0.2, 0.3, 0.3, 1,0.01],[0.1,0.1,0.2,0.01,1]])}


def training_data_reg_all_snr_corrected(total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		SNRdB=np.random.uniform(-4,11,size=(1,))
		SNRdB=float(SNRdB)
		h1, h2 = sample_generation_reg_corrected_all(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def training_data_reg_corrected(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_reg_corrected_all(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y
def sample_generation_reg_corrected_all(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=6, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	mu = np.zeros((number_source,))
	sn = np.random.multivariate_normal(mu, Mapping[number_source], size=(snap_time)).T
	# sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source

def training_data_reg_coherent_colored_noise(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_reg_coherent_fbss_corrected(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def sample_generation_reg_coherent_fbss_corrected(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	number_nonher=int(np.random.randint(1, high=number_source+1, size=(1,),))
	# number_nonher=number_source
	sn=np.zeros((number_source,snap_time),dtype=complex)
	sn[0:number_nonher,:]=(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))
	for coh_i in range(number_nonher,number_source):
		index_coher=int(np.random.randint(0, high=number_nonher, size=(1,),))
		sn[coh_i,:]=sn[index_coher,:]
	# wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	covariance_matrix=np.zeros((number_antenna,number_antenna),dtype=complex)
	for ncov_i in range(number_antenna):
		for ncov_ij in range(number_antenna):
			covariance_matrix[ncov_i,ncov_ij]=(1./np.sqrt(2))*np.power(0.7,np.abs(ncov_i-ncov_ij))*np.exp(1j*np.abs(ncov_i-ncov_ij)*0.77*np.pi)
	L = np.matrix(cholesky(covariance_matrix, lower=True))
	wn=(1./np.sqrt(2))*L.H*(np.random.normal(size=(number_antenna,snap_time))+1j*np.random.normal(size=(number_antenna,snap_time)))

	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	'''
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	'''
	subarray = number_antenna // 2
	Rfor = np.zeros((subarray, subarray), dtype=complex)
	for forlx in range(number_antenna - subarray + 1):
		x_temp = x[forlx:forlx + subarray, :]
		x_temp_her = np.asarray(np.matrix(x_temp).H)
		Rfor += (1 / snap_time) * np.dot(x_temp, x_temp_her) * (1 / (number_antenna - subarray + 1))
	change = np.eye(subarray)

	Rbac = np.asarray(np.matrix(change[:, :: -1]) * np.matrix(np.conj(Rfor)) * np.matrix(change[:, :: -1]))
	Rave = (Rfor + Rbac) / 2
	# Rave=Rfor
	eig_values_D, U = np.linalg.eig(Rave)
	# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
	eig_values = -np.sort(-np.real(eig_values_D))/(10*SNR)

	return eig_values, number_source


def training_data_reg_coherent(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):

		h1, h2 = sample_generation_reg_coherent_fbss(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def training_data_reg_coherent_all_snr(total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		SNRdB=np.random.uniform(-1,31,size=(1,))
		SNRdB=float(SNRdB)
		h1, h2 = sample_generation_reg_coherent_fbss(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y
def sample_generation_reg_coherent_fbss(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	number_nonher=int(np.random.randint(1, high=number_source+1, size=(1,),))
	# number_nonher=number_source
	sn=np.zeros((number_source,snap_time),dtype=complex)
	sn[0:number_nonher,:]=(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))
	for coh_i in range(number_nonher,number_source):
		index_coher=int(np.random.randint(0, high=number_nonher, size=(1,),))
		sn[coh_i,:]=sn[index_coher,:]
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	'''
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	'''
	subarray = number_antenna // 2
	Rfor = np.zeros((subarray, subarray), dtype=complex)
	for forlx in range(number_antenna - subarray + 1):
		x_temp = x[forlx:forlx + subarray, :]
		x_temp_her = np.asarray(np.matrix(x_temp).H)
		Rfor += (1 / snap_time) * np.dot(x_temp, x_temp_her) * (1 / (number_antenna - subarray + 1))
	change = np.eye(subarray)

	Rbac = np.asarray(np.matrix(change[:, :: -1]) * np.matrix(np.conj(Rfor)) * np.matrix(change[:, :: -1]))
	Rave = (Rfor + Rbac) / 2
	# Rave=Rfor
	eig_values_D, U = np.linalg.eig(Rave)
	# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
	eig_values = -np.sort(-np.real(eig_values_D))/(10*SNR)

	return eig_values, number_source

def sample_generation_reg_coherent(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=6, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	number_nonher=int(np.random.randint(1, high=number_source+1, size=(1,),))
	# number_nonher=number_source
	sn=np.zeros((number_source,snap_time),dtype=complex)
	sn[0:number_nonher,:]=(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))
	for coh_i in range(number_nonher,number_source):
		index_coher=int(np.random.randint(0, high=number_nonher, size=(1,),))
		sn[coh_i,:]=sn[index_coher,:]
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source

def training_data_reg_all_snr(total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		SNRdB=np.random.uniform(-4,41,size=(1,))
		SNRdB=float(SNRdB)
		h1, h2 = sample_generation_regression(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def training_data_regression(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_regression(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def training_data_lu_reg(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_regression_lu(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def sample_generation_regression_lu(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	U = cholesky(Rx, lower=False)
	U_norm=np.linalg.norm(U, axis=1, keepdims=False)/(10*SNR)
	return U_norm, number_source

def training_data_eig_lu_eig_class_angle_point(angl,SNRdB,total_samples,number_antenna,snap_time):

	input_eig = []
	input_lu=[]
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		in_eig, in_lu,out_reg,out_cla = sample_generation_eig_lu_reg_class_angle_point(angl,SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		input_lu.append(in_lu)
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass


def sample_generation_eig_lu_reg_class_angle_point(angl,SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	# number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	# number_source=int(number_source)
	number_source=3
	omega_val = [np.sin((angl* nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
	omega_val = np.asarray(omega_val)
	# omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)

	U = cholesky(Rx, lower=False)
	U_norm=np.linalg.norm(U, axis=1, keepdims=False)/(10*SNR)

	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)

	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return eig_values,U_norm, number_source, number_source_label


def training_data_eig_lu_eig_class_all_snr(total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=[]
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		SNRdB=np.random.uniform(0,21,size=(1,))
		SNRdB=float(SNRdB)
		in_eig, in_lu,out_reg,out_cla = sample_generation_eig_lu_reg_class(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		input_lu.append(in_lu)
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass

def training_data_eig_lu_eig_class_coherent_all_snr(total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=[]
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		SNRdB=np.random.randint(0,42,size=(1,))
		SNRdB=float(SNRdB)
		in_eig, in_lu,out_reg,out_cla = sample_generation_eig_lu_reg_class_coherent(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		input_lu.append(in_lu)
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass

def training_data_eig_lu_eig_class_coherent(SNRdB,total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=[]
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		in_eig, in_lu,out_reg,out_cla = sample_generation_eig_lu_reg_class_coherent(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		input_lu.append(in_lu)
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass

def sample_generation_eig_lu_reg_class_coherent(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	# number_source=1
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	number_nonher=int(np.random.randint(1, high=number_source+1, size=(1,),))
	# number_nonher=number_source
	sn=np.zeros((number_source,snap_time),dtype=complex)
	sn[0:number_nonher,:]=(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))
	for coh_i in range(number_nonher,number_source):
		index_coher=int(np.random.randint(0, high=number_nonher, size=(1,),))
		sn[coh_i,:]=sn[index_coher,:]
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn

	subarray = number_antenna // 2
	Rfor = np.zeros((subarray, subarray), dtype=complex)
	for forlx in range(number_antenna - subarray + 1):
		x_temp = x[forlx:forlx + subarray, :]
		x_temp_her = np.asarray(np.matrix(x_temp).H)
		Rfor += (1 / snap_time) * np.dot(x_temp, x_temp_her) * (1 / (number_antenna - subarray + 1))
	change = np.eye(subarray)

	Rbac = np.asarray(np.matrix(change[:, :: -1]) * np.matrix(np.conj(Rfor)) * np.matrix(change[:, :: -1]))
	Rave = (Rfor + Rbac) / 2
	# Rave=Rfor
	U = cholesky(Rave, lower=False)
	U_norm=np.linalg.norm(U, axis=1, keepdims=False)/(10*SNR)

	eig_values_D,_=np.linalg.eig(Rave)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)

	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna//2)
	return eig_values,U_norm, number_source, number_source_label

def training_data_eig_lu_eig_class(SNRdB,total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=[]
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		in_eig, in_lu,out_reg,out_cla = sample_generation_eig_lu_reg_class(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		input_lu.append(in_lu)
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass

def sample_generation_eig_lu_reg_class(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	U = cholesky(Rx, lower=False)
	U_norm=np.linalg.norm(U, axis=1, keepdims=False)/(10*SNR)

	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)

	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return eig_values,U_norm, number_source, number_source_label

def training_data_cov_eig_lu_eig_class(SNRdB,total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=np.array([])
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		in_eig, in_lu,out_reg,out_cla = sample_generation_cov_eig_lu_reg_class(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		# input_lu.append(in_lu)

		if input_lu.size ==0:
			input_lu=in_lu
		elif input_lu.size ==in_lu.size:
			input_lu = np.stack((input_lu, in_lu),axis=0)
		else:
			in_lu = in_lu[np.newaxis, :]
			input_lu = np.vstack((input_lu, in_lu))
		# print(input_lu.shape)


		output_reg.append(out_reg)
		output_class.append(out_cla)
		# print(InputLu.shape)

	InputEig= np.asarray(input_eig)
	# InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,input_lu,OutputReg,OutputClass
def training_data_cov_eig_lu_eig_class_all_snr(total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=np.array([])
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		SNRdB=np.random.uniform(0,42,size=(1,))
		SNRdB=float(SNRdB)
		in_eig, in_lu,out_reg,out_cla = sample_generation_cov_eig_lu_reg_class(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		# input_lu.append(in_lu)

		if input_lu.size ==0:
			input_lu=in_lu
		elif input_lu.size ==in_lu.size:
			input_lu = np.stack((input_lu, in_lu),axis=0)
		else:
			in_lu = in_lu[np.newaxis, :]
			input_lu = np.vstack((input_lu, in_lu))
		# print(input_lu.shape)


		output_reg.append(out_reg)
		output_class.append(out_cla)
		# print(InputLu.shape)

	InputEig= np.asarray(input_eig)
	# InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,input_lu,OutputReg,OutputClass

def sample_generation_cov_eig_lu_reg_class(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)

	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	Rx_full=np.stack((np.real(Rx),np.imag(Rx)),axis=2)/(Power_mag)
	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return eig_values,Rx_full, number_source, number_source_label

def training_data_org_eig_class_coherent_all_snr(total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=np.array([])
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		SNRdB=np.random.uniform(0,42,size=(1,))
		SNRdB=float(SNRdB)
		in_eig, in_lu,out_reg,out_cla = sample_generation_org_reg_class_coherent(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		if input_lu.size == 0:
			input_lu=in_lu
		elif input_lu.size ==in_lu.size:
			input_lu = np.stack((input_lu, in_lu),axis=0)
		else:
			in_lu = in_lu[np.newaxis, :]
			input_lu = np.vstack((input_lu, in_lu))
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass

def training_data_org_eig_class_coherent(SNRdB,total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=np.array([])
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		in_eig, in_lu,out_reg,out_cla = sample_generation_org_reg_class_coherent(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		if input_lu.size == 0:
			input_lu=in_lu
		elif input_lu.size ==in_lu.size:
			input_lu = np.stack((input_lu, in_lu),axis=0)
		else:
			in_lu = in_lu[np.newaxis, :]
			input_lu = np.vstack((input_lu, in_lu))
		output_reg.append(out_reg)
		output_class.append(out_cla)

	InputEig= np.asarray(input_eig)
	InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,InputLu,OutputReg,OutputClass

def sample_generation_org_reg_class_coherent(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	# number_source=1
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	number_nonher=int(np.random.randint(1, high=number_source+1, size=(1,),))
	# number_nonher=number_source
	sn=np.zeros((number_source,snap_time),dtype=complex)
	sn[0:number_nonher,:]=(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_nonher,snap_time))
	for coh_i in range(number_nonher,number_source):
		index_coher=int(np.random.randint(0, high=number_nonher, size=(1,),))
		sn[coh_i,:]=sn[index_coher,:]
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	Rx_full=np.stack((np.real(x),np.imag(x)),axis=2)/(Power_mag)

	subarray = number_antenna // 2
	Rfor = np.zeros((subarray, subarray), dtype=complex)
	for forlx in range(number_antenna - subarray + 1):
		x_temp = x[forlx:forlx + subarray, :]
		x_temp_her = np.asarray(np.matrix(x_temp).H)
		Rfor += (1 / snap_time) * np.dot(x_temp, x_temp_her) * (1 / (number_antenna - subarray + 1))
	change = np.eye(subarray)

	Rbac = np.asarray(np.matrix(change[:, :: -1]) * np.matrix(np.conj(Rfor)) * np.matrix(change[:, :: -1]))
	Rave = (Rfor + Rbac) / 2

	eig_values_D,_=np.linalg.eig(Rave)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)

	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return eig_values,Rx_full, number_source, number_source_label



def training_data_org_eig_lu_eig_class_all_snr(total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=np.array([])
	output_reg=[]
	output_class=[]
	for index_t in range(total_samples):
		SNRdB = np.random.uniform(0, 21, size=(1,))
		SNRdB = float(SNRdB)
		in_eig, in_lu,out_reg,out_cla = sample_generation_org_eig_lu_reg_class(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		# input_lu.append(in_lu)
		if input_lu.size == 0:
			input_lu=in_lu
		elif input_lu.size ==in_lu.size:
			input_lu = np.stack((input_lu, in_lu),axis=0)
		else:
			in_lu = in_lu[np.newaxis, :]
			input_lu = np.vstack((input_lu, in_lu))
		# print(input_lu.shape)


		# print(input_lu.shape)


		output_reg.append(out_reg)
		output_class.append(out_cla)
		# print(InputLu.shape)

	InputEig= np.asarray(input_eig)
	# InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,input_lu,OutputReg,OutputClass


def training_data_org_eig_lu_eig_class(SNRdB,total_samples,number_antenna,snap_time):
	input_eig = []
	input_lu=np.array([])
	output_reg=[]
	output_class=[]

	for index_t in range(total_samples):
		in_eig, in_lu,out_reg,out_cla = sample_generation_org_eig_lu_reg_class(SNRdB,number_antenna,snap_time)
		input_eig.append(in_eig)
		# input_lu.append(in_lu)
		if input_lu.size == 0:
			input_lu=in_lu
		elif input_lu.size ==in_lu.size:
			input_lu = np.stack((input_lu, in_lu),axis=0)
		else:
			in_lu = in_lu[np.newaxis, :]
			input_lu = np.vstack((input_lu, in_lu))
		# print(input_lu.shape)


		# print(input_lu.shape)


		output_reg.append(out_reg)
		output_class.append(out_cla)
		# print(InputLu.shape)

	InputEig= np.asarray(input_eig)
	# InputLu= np.asarray(input_lu)
	OutputReg = np.asarray(output_reg).reshape((total_samples,1))
	OutputClass= np.asarray(output_class)
	return InputEig,input_lu,OutputReg,OutputClass

def sample_generation_org_eig_lu_reg_class(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)

	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)

	Rx_full=np.stack((np.real(x),np.imag(x)),axis=2)/(Power_mag)

	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return eig_values,Rx_full, number_source, number_source_label


def training_data_eig_reg(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_regression(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y

def sample_generation_regression(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source


def training_data_reg_source_angle_same(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_reg_source_angle_same(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y


def sample_generation_reg_source_angle_same(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source = 3
	omega_val = [np.sin((5.5 * nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
	omega_val = np.asarray(omega_val)

	# number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	# number_source=int(number_source)
	# omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source

def training_data_reg_angle_varying(angl,SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_reg_angle_varying(angl,SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y


def sample_generation_reg_angle_varying(angl,SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val = [np.sin((angl* nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
	omega_val = np.asarray(omega_val)
	# omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source


def sample_generation_reg_corrected(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=6, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0,math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	mu = np.zeros((number_source,))
	sn = np.random.multivariate_normal(mu, Mapping[number_source], size=(snap_time)).T
	# sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source







#
# import scipy.io as scio
#
# dataFile = 'test.mat'
# data = scio.loadmat(dataFile)
# sn=data['sn']
# wn=data['wn']
# A=data['A']

def MDL_MMSE(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=100000
	acc=0
	acc_mmse=0
	acc_aic=0
	for rept in range(reat_time):
		# number_source=3
		# omega_val=[np.sin((5.5*nn/180)*np.pi)*0.5*2*np.pi for nn in range(number_source)]
		# omega_val=np.asarray(omega_val)
		# omega_val=np.array([0,np.sin((5.5/180)*np.pi)*0.5*2*np.pi,np.sin((11./180)*np.pi)*0.5*2*np.pi])

		if rept% 50==0:
			print(rept)
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		# number_source=3
		omega_val=np.random.uniform(0,math.pi, (number_source,1))
		# omega_val = [np.sin((5.5 * nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
		# omega_val = np.asarray(omega_val)

		A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
		sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		# '''MMSE　MDL'''
		# dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		# r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		# r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		# U_sorted=U[:, np.argsort(-eig_values_D)]
		# r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		# r_dx_U=r_dx_U.T
		# r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		# MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		# est_i =np.arange(1,number_antenna+1)
		# est_i = np.reshape(est_i,(np.size(est_i),1))
		# # print(np.min(MMSE_chi))
		# MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		# MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		# k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		# if k_esti_mmse==number_source:
		# 	acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''AIC'''
		AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
			np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
							   2 * number_antenna - k) for k in range(number_antenna)]
		k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		if k_esti_aic==number_source:
			acc_aic+=1./reat_time
	return acc,acc_mmse,acc_aic

def MDL_subfun_random_angle(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	acc_mmse=0
	acc_aic=0
	for rept in range(reat_time):
		# number_source=3
		# omega_val=[np.sin((5.5*nn/180)*np.pi)*0.5*2*np.pi for nn in range(number_source)]
		# omega_val=np.asarray(omega_val)
		# omega_val=np.array([0,np.sin((5.5/180)*np.pi)*0.5*2*np.pi,np.sin((11./180)*np.pi)*0.5*2*np.pi])
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		omega_val=np.random.uniform(0,math.pi, (number_source,1))
		# omega_val = [np.sin((5.5 * nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
		# omega_val = np.asarray(omega_val)
		A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
		sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		'''MMSE　MDL'''
		dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		U_sorted=U[:, np.argsort(-eig_values_D)]
		r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		r_dx_U=r_dx_U.T
		r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		est_i =np.arange(1,number_antenna+1)
		est_i = np.reshape(est_i,(np.size(est_i),1))
		# print(np.min(MMSE_chi))
		MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		if k_esti_mmse==number_source:
			acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''AIC'''
		AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
			np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
							   2 * number_antenna - k) for k in range(number_antenna)]
		k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		if k_esti_aic==number_source:
			acc_aic+=1./reat_time

	return acc,acc_mmse,acc_aic

def MDL_fbss_random_angle_coherent_colored(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	acc_mmse=0
	acc_fbss_mdl=0
	for rept in range(reat_time):
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		omega_val = np.random.uniform(0,  math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		number_nonher = int(np.random.randint(1, high=number_source + 1, size=(1,), ))
		sn = np.zeros((number_source, snap_time), dtype=complex)
		sn[0:number_nonher, :] = (1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time)) + 1j * (
					1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time))
		for coh_i in range(number_nonher, number_source):
			index_coher = int(np.random.randint(0, high=number_nonher, size=(1,), ))
			sn[coh_i, :] = sn[index_coher, :]

		# wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))

		covariance_matrix = np.zeros((number_antenna, number_antenna), dtype=complex)
		for ncov_i in range(number_antenna):
			for ncov_ij in range(number_antenna):
				covariance_matrix[ncov_i, ncov_ij] = (1. / np.sqrt(2)) * np.power(0.7,
																				  np.abs(ncov_i - ncov_ij)) * np.exp(
					1j * np.abs(ncov_i - ncov_ij) * 0.77 * np.pi)
		L = np.matrix(cholesky(covariance_matrix, lower=True))
		wn = (1. / np.sqrt(2)) * L.H * (np.random.normal(size=(number_antenna, snap_time)) + 1j * np.random.normal(
			size=(number_antenna, snap_time)))

		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		'''MMSE　MDL'''
		dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		U_sorted=U[:, np.argsort(-eig_values_D)]
		r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		r_dx_U=r_dx_U.T
		r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		est_i =np.arange(1,number_antenna+1)
		est_i = np.reshape(est_i,(np.size(est_i),1))
		# print(np.min(MMSE_chi))
		MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		if k_esti_mmse==number_source:
			acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''FBSS MDL'''
		subarray = number_antenna // 2
		Rfor = np.zeros((subarray, subarray),dtype=complex)
		for forlx in range(number_antenna - subarray + 1):
			x_temp=x[forlx:forlx+subarray,:]
			x_temp_her = np.asarray(np.matrix(x_temp).H)
			Rfor += (1 / snap_time)*np.dot(x_temp,x_temp_her)*(1/(number_antenna-subarray+1))
		change = np.eye(subarray)

		Rbac = np.asarray(np.matrix(change[:, :: -1])*np.matrix(np.conj(Rfor))* np.matrix(change[:, :: -1]))
		Rave = (Rfor + Rbac) / 2
		# Rave=Rfor
		eig_values_D,U=np.linalg.eig(Rave)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(subarray)]
		k_esti_fbss_mdl=MDL_fun.index(np.min(MDL_fun))
		if k_esti_fbss_mdl==number_source:
			acc_fbss_mdl+=1./reat_time

	return acc,acc_mmse,acc_fbss_mdl
# def MDL_fbss_random_angle_coherent_colored_corrected(SNRdB,number_antenna,snap_time):
#
# 	SNR = 10 ** (SNRdB / 10)
# 	reat_time=2000
# 	acc=0
# 	acc_corrected=0
# 	acc_fbss_mdl=0
# 	acc_fbss_mdl_corrected=0
# 	for rept in range(reat_time):
# 		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
# 		number_source=int(number_source)
# 		omega_val = np.random.uniform(0,  math.pi, (number_source, 1))
# 		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
# 							   omega_val.reshape((1, number_source))))
# 		number_nonher = int(np.random.randint(1, high=number_source + 1, size=(1,), ))
# 		sn = np.zeros((number_source, snap_time), dtype=complex)
# 		sn[0:number_nonher, :] = (1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time)) + 1j * (
# 					1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time))
# 		for coh_i in range(number_nonher, number_source):
# 			index_coher = int(np.random.randint(0, high=number_nonher, size=(1,), ))
# 			sn[coh_i, :] = sn[index_coher, :]
#
# 		# wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
#
# 		covariance_matrix = np.zeros((number_antenna, number_antenna), dtype=complex)
# 		for ncov_i in range(number_antenna):
# 			for ncov_ij in range(number_antenna):
# 				covariance_matrix[ncov_i, ncov_ij] = (1. / np.sqrt(2)) * np.power(0.7,
# 																				  np.abs(ncov_i - ncov_ij)) * np.exp(
# 					1j * np.abs(ncov_i - ncov_ij) * 0.77 * np.pi)
# 		L = np.matrix(cholesky(covariance_matrix, lower=True))
# 		wn = (1. / np.sqrt(2)) * L.H * (np.random.normal(size=(number_antenna, snap_time)) + 1j * np.random.normal(
# 			size=(number_antenna, snap_time)))
#
# 		Power_mag = np.sqrt(SNR)
# 		x =  np.dot(A,sn) + (1./Power_mag)*wn
# 		x_her=np.asarray(np.matrix(x).H)
# 		Rx=(1./snap_time)*np.dot(x,x_her)
# 		eig_values_D,U=np.linalg.eig(Rx)
# 		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
# 		eig_values=-np.sort(-np.real(eig_values_D))
# 		# '''MMSE　MDL'''
# 		# dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
# 		# r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
# 		# r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
# 		# U_sorted=U[:, np.argsort(-eig_values_D)]
# 		# r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
# 		# r_dx_U=r_dx_U.T
# 		# r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
# 		# MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
# 		# est_i =np.arange(1,number_antenna+1)
# 		# est_i = np.reshape(est_i,(np.size(est_i),1))
# 		# # print(np.min(MMSE_chi))
# 		# MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
# 		# MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
# 		# k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
# 		# if k_esti_mmse==number_source:
# 		# 	acc_mmse+=1./reat_time
#
# 		'''MDL'''
#
# 		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
# 		k_esti=MDL_fun.index(np.min(MDL_fun))
# 		if k_esti==number_source:
# 			acc+=1./reat_time
#
# 		'''MDL corrected'''
# 		eig_values_revise=np.sqrt(np.cumsum(eig_values))
# 		eig_values=eig_values+eig_values_revise
# 		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
# 		k_esti=MDL_fun.index(np.min(MDL_fun))
# 		if k_esti==number_source:
# 			acc_corrected+=1./reat_time
#
# 		'''FBSS MDL'''
# 		subarray = number_antenna // 2
# 		Rfor = np.zeros((subarray, subarray),dtype=complex)
# 		for forlx in range(number_antenna - subarray + 1):
# 			x_temp=x[forlx:forlx+subarray,:]
# 			x_temp_her = np.asarray(np.matrix(x_temp).H)
# 			Rfor += (1 / snap_time)*np.dot(x_temp,x_temp_her)*(1/(number_antenna-subarray+1))
# 		change = np.eye(subarray)
#
# 		Rbac = np.asarray(np.matrix(change[:, :: -1])*np.matrix(np.conj(Rfor))* np.matrix(change[:, :: -1]))
# 		Rave = (Rfor + Rbac) / 2
# 		# Rave=Rfor
# 		eig_values_D,U=np.linalg.eig(Rave)
# 		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
# 		eig_values=-np.sort(-np.real(eig_values_D))
#
# 		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(subarray)]
# 		k_esti_fbss_mdl=MDL_fun.index(np.min(MDL_fun))
# 		if k_esti_fbss_mdl==number_source:
# 			acc_fbss_mdl+=1./reat_time
#
#
# 		'''FBSS MDL corrected'''
# 		eig_values_revise=np.sqrt(np.cumsum(eig_values))
# 		eig_values=eig_values+eig_values_revise
#
# 		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(subarray)]
# 		k_esti_fbss_mdl=MDL_fun.index(np.min(MDL_fun))
# 		if k_esti_fbss_mdl==number_source:
# 			acc_fbss_mdl_corrected+=1./reat_time
#
# 	return acc,acc_fbss_mdl,acc_corrected,acc_fbss_mdl_corrected

def MDL_fbss_random_angle_coherent_colored_corrected(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	acc_corrected=0
	acc_fbss_mdl=0
	for rept in range(reat_time):
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		omega_val = np.random.uniform(0,  math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		number_nonher = int(np.random.randint(1, high=number_source + 1, size=(1,), ))
		sn = np.zeros((number_source, snap_time), dtype=complex)
		sn[0:number_nonher, :] = (1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time)) + 1j * (
					1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time))
		for coh_i in range(number_nonher, number_source):
			index_coher = int(np.random.randint(0, high=number_nonher, size=(1,), ))
			sn[coh_i, :] = sn[index_coher, :]

		# wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))

		covariance_matrix = np.zeros((number_antenna, number_antenna), dtype=complex)
		for ncov_i in range(number_antenna):
			for ncov_ij in range(number_antenna):
				covariance_matrix[ncov_i, ncov_ij] = (1. / np.sqrt(2)) * np.power(0.7,
																				  np.abs(ncov_i - ncov_ij)) * np.exp(
					1j * np.abs(ncov_i - ncov_ij) * 0.77 * np.pi)
		L = np.matrix(cholesky(covariance_matrix, lower=True))
		wn = (1. / np.sqrt(2)) * L.H * (np.random.normal(size=(number_antenna, snap_time)) + 1j * np.random.normal(
			size=(number_antenna, snap_time)))

		Power_mag = np.sqrt(SNR)
		x =  Power_mag*np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))


		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''MDL corrected'''
		eig_values_revise=np.sqrt(np.cumsum(eig_values))
		eig_values=eig_values+eig_values_revise
		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc_corrected+=1./reat_time

		'''FBSS MDL'''
		subarray = number_antenna // 2
		Rfor = np.zeros((subarray, subarray),dtype=complex)
		for forlx in range(number_antenna - subarray + 1):
			x_temp=x[forlx:forlx+subarray,:]
			x_temp_her = np.asarray(np.matrix(x_temp).H)
			Rfor += (1 / snap_time)*np.dot(x_temp,x_temp_her)*(1/(number_antenna-subarray+1))
		change = np.eye(subarray)

		Rbac = np.asarray(np.matrix(change[:, :: -1])*np.matrix(np.conj(Rfor))* np.matrix(change[:, :: -1]))
		Rave = (Rfor + Rbac) / 2
		# Rave=Rfor
		eig_values_D,U=np.linalg.eig(Rave)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(subarray)]
		k_esti_fbss_mdl=MDL_fun.index(np.min(MDL_fun))
		if k_esti_fbss_mdl==number_source:
			acc_fbss_mdl+=1./reat_time

	return acc,acc_corrected,acc_fbss_mdl,0

def MDL_fbss_random_angle_coherent(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=10000
	acc=0
	acc_mmse=0
	acc_fbss_mdl=0
	acc_aic_fbss=0
	for rept in range(reat_time):
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		omega_val = np.random.uniform(0, math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		number_nonher = int(np.random.randint(1, high=number_source + 1, size=(1,), ))
		sn = np.zeros((number_source, snap_time), dtype=complex)
		sn[0:number_nonher, :] = (1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time)) + 1j * (
					1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time))
		for coh_i in range(number_nonher, number_source):
			index_coher = int(np.random.randint(0, high=number_nonher, size=(1,), ))
			sn[coh_i, :] = sn[index_coher, :]

		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		# '''MMSE　MDL'''
		# dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		# r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		# r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		# U_sorted=U[:, np.argsort(-eig_values_D)]
		# r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		# r_dx_U=r_dx_U.T
		# r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		# MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		# est_i =np.arange(1,number_antenna+1)
		# est_i = np.reshape(est_i,(np.size(est_i),1))
		# # print(np.min(MMSE_chi))
		# MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		# MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		# k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		# if k_esti_mmse==number_source:
		# 	acc_mmse+=1./reat_time
		#
		# '''MDL'''
		#
		# MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		# k_esti=MDL_fun.index(np.min(MDL_fun))
		# if k_esti==number_source:
		# 	acc+=1./reat_time

		'''FBSS MDL'''
		subarray = number_antenna // 2
		Rfor = np.zeros((subarray, subarray),dtype=complex)
		for forlx in range(number_antenna - subarray + 1):
			x_temp=x[forlx:forlx+subarray,:]
			x_temp_her = np.asarray(np.matrix(x_temp).H)
			Rfor += (1 / snap_time)*np.dot(x_temp,x_temp_her)*(1/(number_antenna-subarray+1))
		change = np.eye(subarray)

		Rbac = np.asarray(np.matrix(change[:, :: -1])*np.matrix(np.conj(Rfor))* np.matrix(change[:, :: -1]))
		Rave = (Rfor + Rbac) / 2
		# Rave=Rfor
		eig_values_D,U=np.linalg.eig(Rave)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(subarray)]
		k_esti_fbss_mdl=MDL_fun.index(np.min(MDL_fun))
		if k_esti_fbss_mdl==number_source:
			acc_fbss_mdl+=1./reat_time


		'''AIC'''
		AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
			np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
							   2 * number_antenna - k) for k in range(subarray)]
		k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		if k_esti_aic==number_source:
			acc_aic_fbss+=1./reat_time

	return acc_fbss_mdl,acc_aic_fbss

def MDL_subfun_random_angle_coherent(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	acc_mmse=0
	acc_aic=0
	for rept in range(reat_time):
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		omega_val = np.random.uniform(0,  math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		number_nonher = int(np.random.randint(1, high=number_source + 1, size=(1,), ))
		sn = np.zeros((number_source, snap_time), dtype=complex)
		sn[0:number_nonher, :] = (1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time)) + 1j * (
					1. / np.sqrt(2)) * np.random.normal(size=(number_nonher, snap_time))
		for coh_i in range(number_nonher, number_source):
			index_coher = int(np.random.randint(0, high=number_nonher, size=(1,), ))
			sn[coh_i, :] = sn[index_coher, :]

		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		'''MMSE　MDL'''
		dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		U_sorted=U[:, np.argsort(-eig_values_D)]
		r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		r_dx_U=r_dx_U.T
		r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		est_i =np.arange(1,number_antenna+1)
		est_i = np.reshape(est_i,(np.size(est_i),1))
		# print(np.min(MMSE_chi))
		MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		if k_esti_mmse==number_source:
			acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''AIC'''
		AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
			np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
							   2 * number_antenna - k) for k in range(number_antenna)]
		k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		if k_esti_aic==number_source:
			acc_aic+=1./reat_time

	return acc,acc_mmse,acc_aic

def MDL_subfun_random_angle_corrected(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	acc_mmse=0
	acc_aic=0
	for rept in range(reat_time):
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		omega_val = np.random.uniform(0,   math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		mu = np.zeros((number_source,))
		sn = np.random.multivariate_normal(mu, Mapping[number_source], size=(snap_time)).T
		# # number_source=3
		# # omega_val=[np.sin((5.5*nn/180)*np.pi)*0.5*2*np.pi for nn in range(number_source)]
		# # omega_val=np.asarray(omega_val)
		# # omega_val=np.array([0,np.sin((5.5/180)*np.pi)*0.5*2*np.pi,np.sin((11./180)*np.pi)*0.5*2*np.pi])
		# number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		# number_source=int(number_source)
		# omega_val=np.random.uniform(0, math.pi, (number_source,1))
		# # omega_val = [np.sin((5.5 * nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
		# # omega_val = np.asarray(omega_val)
		# A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
		# sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		'''MMSE　MDL'''
		dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		U_sorted=U[:, np.argsort(-eig_values_D)]
		r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		r_dx_U=r_dx_U.T
		r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		est_i =np.arange(1,number_antenna+1)
		est_i = np.reshape(est_i,(np.size(est_i),1))
		# print(np.min(MMSE_chi))
		MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		if k_esti_mmse==number_source:
			acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''AIC'''
		AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
			np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
							   2 * number_antenna - k) for k in range(number_antenna)]
		k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		if k_esti_aic==number_source:
			acc_aic+=1./reat_time

	return acc,acc_mmse,acc_aic


def MDL_subfun_vs_angle(angle,SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=10000
	acc=0
	acc_mmse=0
	acc_aic=0
	for rept in range(reat_time):
		# number_source=3
		# omega_val=[np.sin((5.5*nn/180)*np.pi)*0.5*2*np.pi for nn in range(number_source)]
		# omega_val=np.asarray(omega_val)
		# omega_val=np.array([0,np.sin((5.5/180)*np.pi)*0.5*2*np.pi,np.sin((11./180)*np.pi)*0.5*2*np.pi])

		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)

		# number_source=2
		# omega_val=np.random.uniform(0, math.pi, (number_source,1))
		omega_val = [np.sin((angle * nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
		omega_val = np.asarray(omega_val)
		A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
		sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		# '''MMSE　MDL'''
		dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		U_sorted=U[:, np.argsort(-eig_values_D)]
		r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		r_dx_U=r_dx_U.T
		r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		est_i =np.arange(1,number_antenna+1)
		est_i = np.reshape(est_i,(np.size(est_i),1))
		MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		if k_esti_mmse==number_source:
			acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		# '''AIC'''
		# AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
		# 	np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
		# 					   2 * number_antenna - k) for k in range(number_antenna)]
		# k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		# if k_esti_aic==number_source:
		# 	acc_aic+=1./reat_time

	return acc,acc_mmse,acc_aic

def MDL_subfun(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	acc_mmse=0
	acc_aic=0
	for rept in range(reat_time):
		# number_source=3
		# omega_val=[np.sin((5.5*nn/180)*np.pi)*0.5*2*np.pi for nn in range(number_source)]
		# omega_val=np.asarray(omega_val)
		# omega_val=np.array([0,np.sin((5.5/180)*np.pi)*0.5*2*np.pi,np.sin((11./180)*np.pi)*0.5*2*np.pi])
		number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
		number_source=int(number_source)
		# omega_val=np.random.uniform(0, math.pi, (number_source,1))
		omega_val = [np.sin((5.5 * nn / 180) * np.pi) * 0.5 * 2 * np.pi for nn in range(number_source)]
		omega_val = np.asarray(omega_val)
		A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
		sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,U=np.linalg.eig(Rx)
		# np.matrix(eig_matrix)*np.diag(eig_values_D)*np.matrix(eig_matrix).H-np.matrix(Rx)=0
		eig_values=-np.sort(-np.real(eig_values_D))
		'''MMSE　MDL'''
		dn=np.reshape(sn[0,:],newshape=(np.size(sn[0,:]),1))
		r_dx=np.mean(np.matrix(x)*np.matrix(np.diag(sn[0,:])).H,1)
		r_dx=np.reshape(r_dx,newshape=(np.size(r_dx),1))
		U_sorted=U[:, np.argsort(-eig_values_D)]
		r_dx_U=np.matrix(r_dx).H*np.matrix(U_sorted)
		r_dx_U=r_dx_U.T
		r_dx_U_eig=np.power(np.abs(r_dx_U),2)/eig_values.reshape((np.size(eig_values),1))
		MMSE_chi= (1 / snap_time) * (np.matrix(dn).H*np.matrix(dn))-np.cumsum(r_dx_U_eig,axis=0)
		est_i =np.arange(1,number_antenna+1)
		est_i = np.reshape(est_i,(np.size(est_i),1))
		MMSE_MDL_fun_pre = snap_time * np.log(MMSE_chi) + 0.5 * (np.power(est_i,2)+ est_i) * np.log(snap_time)
		MMSE_MDL_fun=np.ravel(np.real(MMSE_MDL_fun_pre)).tolist()
		k_esti_mmse=MMSE_MDL_fun.index(np.min(MMSE_MDL_fun))+1
		if k_esti_mmse==number_source:
			acc_mmse+=1./reat_time

		'''MDL'''

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(np.min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time

		'''AIC'''
		AIC_fun = [-2*snap_time * np.log((np.prod(eig_values[k])) / (
			np.power(((1. / (number_antenna - k)) * np.sum(eig_values[k])), number_antenna - k))) + 2* k * (
							   2 * number_antenna - k) for k in range(number_antenna)]
		k_esti_aic = AIC_fun.index(np.min(AIC_fun))
		if k_esti_aic==number_source:
			acc_aic+=1./reat_time

	return acc,acc_mmse,acc_aic

def MDL_subfun_correced(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	for rept in range(reat_time):

		number_source = 2
		omega_val = np.random.uniform(0,   math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		mu = np.zeros((number_source,))
		Sig = np.array([[0.2, 0.4], [0.4, 1]])
		sn = np.random.multivariate_normal(mu, Sig, size=(snap_time)).T

		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,_=np.linalg.eig(Rx)
		eig_values=-np.sort(-np.real(eig_values_D))

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time
	return acc


def MDL_subfun_correced_all(SNRdB,number_antenna,snap_time):

	SNR = 10 ** (SNRdB / 10)
	reat_time=2000
	acc=0
	for rept in range(reat_time):
		number_source=np.random.randint(1, high=6, size=(1,),)
		number_source=int(number_source)
		omega_val = np.random.uniform(0,  math.pi, (number_source, 1))
		A = np.exp(1j * np.dot(np.arange(0, number_antenna, 1).reshape((number_antenna, 1)),
							   omega_val.reshape((1, number_source))))
		mu = np.zeros((number_source,))
		sn = np.random.multivariate_normal(mu, Mapping[number_source], size=(snap_time)).T

		wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
		Power_mag = np.sqrt(SNR)
		x = Power_mag * np.dot(A,sn) + wn
		x_her=np.asarray(np.matrix(x).H)
		Rx=(1./snap_time)*np.dot(x,x_her)
		eig_values_D,_=np.linalg.eig(Rx)
		eig_values=-np.sort(-np.real(eig_values_D))

		MDL_fun=[-snap_time*np.log((np.prod(eig_values[k]))/(np.power(((1. / (number_antenna - k) )*np.sum(eig_values[k])),number_antenna-k)))+0.5 * k * (2 * number_antenna - k) * np.log(snap_time) for k in range(number_antenna)]
		k_esti=MDL_fun.index(min(MDL_fun))
		if k_esti==number_source:
			acc+=1./reat_time
	return acc


def training_data_lu_class(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_class_lu(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels)
	return batch_x,batch_y

def sample_generation_class_lu(SNRdB,number_antenna,snap_time):
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0, math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	U = cholesky(Rx, lower=False)
	U_norm=np.linalg.norm(U, axis=1, keepdims=False)/(10*SNR)
	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return U_norm, number_source_label

def training_data_eig_class(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_class(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels)
	return batch_x,batch_y


def training_data_class(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_class(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels)
	return batch_x,batch_y

def sample_generation_class(SNRdB,number_antenna,snap_time):
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0, math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x = Power_mag * np.dot(A,sn) + wn
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10*SNR)
	number_source_label = np_utils.to_categorical(number_source, num_classes=number_antenna)
	return eig_values, number_source_label


def training_data_class_lu(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_class_lu(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels)
	return batch_x,batch_y


def training_data_regression_norm(SNRdB,total_samples,number_antenna,snap_time):
	input_samples = []
	input_labels = []

	for index_t in range(total_samples):
		h1, h2 = sample_generation_regression(SNRdB,number_antenna,snap_time)
		input_labels.append(h2)
		input_samples.append(h1)
	batch_x = np.asarray(input_samples)
	batch_y = np.asarray(input_labels).reshape((total_samples,1))
	return batch_x,batch_y


def sample_generation_regression_norm(SNRdB,number_antenna,snap_time):
	# SNRdB = 20.
	SNR = 10 ** (SNRdB / 10)
	number_source=np.random.randint(1, high=number_antenna//2, size=(1,),)
	number_source=int(number_source)
	omega_val=np.random.uniform(0, math.pi, (number_source,1))
	A=np.exp(1j*np.dot(np.arange(0,number_antenna,1).reshape((number_antenna,1)),omega_val.reshape((1,number_source))))
	sn=(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_source,snap_time))
	wn=(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))+1j*(1./np.sqrt(2))*np.random.normal(size=(number_antenna,snap_time))
	Power_mag = np.sqrt(SNR)
	x =  np.dot(A,sn) + wn/Power_mag
	x_her=np.asarray(np.matrix(x).H)
	Rx=(1./snap_time)*np.dot(x,x_her)
	eig_values_D,_=np.linalg.eig(Rx)
	eig_values=-np.sort(-np.real(eig_values_D))/(10)
	# eig_values_scaled = preprocessing.scale(eig_values)
	# eig_values_scaled = (eig_values)
	return eig_values, number_source


def draw_image(mean_error_train):
	ber_rate_train = np.asarray(mean_error_train)
	ber_rate_test_result = ber_rate_train[len(ber_rate_train) - 30:len(ber_rate_train)]
	NMSE_mean=np.mean(ber_rate_test_result)
	print("test ber mean:", NMSE_mean)
	sio.savemat('accuracy.mat', {'BER_train': ber_rate_train,'NMSE_mean':NMSE_mean})

	ber_epoch = np.linspace(0, len(ber_rate_train) - 1, len(ber_rate_train))
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	# plt.semilogy(ber_epoch, ber_rate_test, "+-", label="ber_rate_test")
	plt.semilogy(ber_epoch, ber_rate_train, "*-", label="accuracy")
	plt.grid(True)
	plt.show()
