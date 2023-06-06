import math as mt
import numpy as np
import os.path
import copy
import sys

import matplotlib.pylab as plt

import codecs,json

import BoltzTraP2.dft as BTP
import BoltzTraP2.bandlib as BL
import BoltzTraP2.io as IO
from BoltzTraP2 import sphere
from BoltzTraP2 import fite
from BoltzTraP2 import serialization
from BoltzTraP2.misc import ffloat
from BoltzTraP2 import units
import multiprocessing as mp

import relaxTau

# funzione per ricavare l'indice del vettore k non usare questo per VASP
def indK(k,dim):
	"""Do not use this. use indK_inv.
	Gives the index of a k-point.
	Input: (k-point, vector dims)
	output: integer
	vector dims is a vector with the number of divisions along each axis in reciprocal lattice"""
	return (mt.floor(dim[0]/2.) + k[0])*dim[1]*dim[2] + (mt.floor(dim[1]/2.) + k[1])*dim[2] + mt.floor(dim[2]/2.) +k[2]

# funzione per ricavare l'indice del vettore k prima a poi b poi c
def indK_inv(k,dim):
	"""Gives the index of a k-point.
	Input: (k-point, vector dims)
	output: integer
	vector dims is a vector with the number of divisions along each axis in reciprocal lattice"""
	return 		(mt.floor(dim[2]/2.) + k[2])*dim[0]*dim[1] + (mt.floor(dim[1]/2.) + k[1])*dim[0] + (mt.floor(dim[0]/2.) +k[0])

def getDimsKpoints(equivalences):
	"""Gives vector of dimensions and list of all kpoints in fine grid.
	Input: equivalences
	Output: dims,k-points
	 """
	# questo serve per definire le dimensioni delle bande, cioè quanti k-punti totali servono.
	dallvec = np.vstack(equivalences)
	#print('questo è dallvec, lungo:',len(dallvec),'\n',dallvec)
	dims = 2 * np.max(np.abs(dallvec), axis=0) + 1
	#print('questo è dims:\n', dims)

	# crea una lista dei k-points totali, quelli moltiplicati per mult
	kpnts = []
	for i in range(-mt.floor(dims[0]/2.), mt.floor(dims[0]/2.)+1):
		for j in range(-mt.floor(dims[1]/2.), mt.floor(dims[1]/2.)+1):
			for k in range(-mt.floor(dims[2]/2.), mt.floor(dims[2]/2.)+1):
				kpnts.append([i,j,k])
	kpnts = np.asarray(kpnts)# crea una lista dei k-points
	
	print('computed dims and k-points')
	return dims,kpnts
	
		


def Tau_bkTmu(Tr,i_T, e_bk,datata,latVec,tau_DFT):
	"""
	compute tau_bk for a given T and mu
	input: 	
		- array of all temperatures
		- index of temperature you want tau_bkT for
		- array of all chemical potentials
		- specific chemical potential
		- e_bk energy bands 
		- datatau where tau for dft is stored
		- latVec direct lattice vectors of system (from Vasp data)
	output:
		- tau_bk evaluated at the given T
	"""
	# we interpolate tau_bk the same way we interpolated e_bk 
	tau_bkTmu = np.zeros((len(e_bk), len(e_bk[0])))
	
	dat = copy.deepcopy(datata)
	dat.ebands = tau_DFT[i_T]
	coef_tau = fite.fitde3D(dat,equivalences)
	tau_bkTmu = fite.getBTPbands(equivalences, coef_tau, latVec, curvature=False, nworkers=6)[0]
	
	tau_bkTmu = np.asarray(tau_bkTmu)
	
	return tau_bkTmu

# get minimum energy of conduction band E_min_CB
def E_min_CB(ebk,fermi0,dos):
	"""
	gives the minimum energy of conduction band. if offset and skimming is applied before it gives the next smallest value of energy from CB.
	Input: 	- list of energies e_bk
			- fermi energy
			- list of density of states
	output flaot minimum from CB
	"""
	b1=0
	cond=1
	while (cond):
		cond0 = ebk[b1] <= fermi0
		cond = np.prod(cond0) 
		b1+=1
	b1 -= 1
	
	#auxiliary = ene[ np.logical_and(ene > fermi0 , dos!=0) ]
	
	return np.amin(ebk[b1])

def skim_ebk(ebk,vxvbk,tmax):
	"""
	skim all bands which are totally negative, i.e. skim all valence bands. returns ndarray of only the conduction bands.
	
	input: 	
		- e_bk ndarray of all bands and kpoints
		- vxv_bk ndarray with number of bands on axis 0
		- max temperature of simulation. Used to skim too high energies which are modulated by fermi function into integrals
	output: 
		- vxv_bk skimmed of only the valence bands
	
	it can be used also like (ebk,ebk) so to get only conduction bands energies
	"""
	b1=0
	cond=1
	while (cond):
		cond0 = ebk[b1] <= 0
		cond = np.prod(cond0) 
		b1+=1
	b1 -= 1
	
	b2=0
	cond=1
	while (cond and (b2<len(ebk))):
		cond0 = ebk[b2] < 0.06
		cond = np.prod(cond0) 
		b2+=1
	b2 = b2-1
	
	
	vxv_skim = []
	for ib in range(b1,b2):
		vxv_skim.append(vxvbk[ib])
	
	vxv_skim = np.array(vxv_skim)
	
	return vxv_skim

def tau_dft_T(Tr,val_mu,datat,que):
	"""
	compute tau_dft(T,mu) for a given array of temperatures and a fixed chemical potential. It is structured for multiprocessing with several sub-arrays of temperature working simultaneously.
	inputs: - array of temperatures
			- array of chemical potentials
			- data structure, the same as data class for btp2
			- queue for multiprocessing
	outputs: - tau_dft(T,mu)
	"""
	tau_dft = np.zeros((len(Tr), len(datat.ebands), len(datat.kpoints)))
	for i_T,val_T in enumerate(Tr):
		for i_b in range(len(datat.ebands)):
			for i_k in range(len(datat.kpoints)):
				en_bk = datat.ebands[i_b,i_k]
				if(en_bk <= 0):
					tau_dft[i_T,i_b,i_k] = 0
				else:
					tau_dft[i_T,i_b,i_k] = relaxTau.tau_tot(val_T , val_mu/units.Joule , en_bk/units.Joule , 1 )
		print('computed tau_dft for T={}'.format(val_T))
	que.put(tau_dft)
	
#def tau_T(Tr,val_mu,ebk,que):
	#"""
	#compute tau(T,mu) for a given array of temperatures and a fixed chemical potential. It is structured for multiprocessing with several sub-arrays of temperature working simultaneously. use atomic units.
	#inputs: 
		#- array of temperatures
		#- value of chemical potential
		#- e_bk
		#- queue for multiprocessing
	#outputs: 
		#- tau_dft(T,mu)
	#"""
	#tau = np.zeros((len(Tr), len(ebk), len(ebk[0])))
	#for i_T,val_T in enumerate(Tr):
		#for i_b in range(len(ebk)):
			##offsetting ebk so that for each band zero is the minimum energy of that band
			#ebk = ebk - np.amin(ebk[i_b])
			#val_mu = val_mu - np.amin(ebk[i_b])
			#for i_k in range(len(ebk[0])):
				## se l'energia è nulla allora gli do un valore molto piccolo, vicino allo zero.
				#if(ebk[i_b,i_k] == 0):
					##tau[i_T,i_b,i_k] = 0 
					#tau[i_T,i_b,i_k] = relaxTau.tau_tot(val_T, val_mu/units.Joule, 1e-6/units.Joule,1)
				#else:
					#tau[i_T,i_b,i_k] = relaxTau.tau_tot(val_T, val_mu/units.Joule, ebk[i_b,i_k]/units.Joule,1 )
		#print('computed tau for T={}'.format(val_T))
	#que.put(tau)
	
	
def tau_T(Tr,val_mu,ebk,dop,rep,que):
	"""
	compute tau(T,mu) for a given array of temperatures and a fixed chemical potential. It is structured for multiprocessing with several sub-arrays of temperature working simultaneously. use atomic units.
	inputs: 
		- array of temperatures
		- value of chemical potential
		- e_bk
		- queue for multiprocessing
	outputs: 
		- tau_dft(T,mu)
	"""
	tau = np.zeros((len(Tr), len(ebk), len(ebk[0])))
	for i_T,val_T in enumerate(Tr):
		for i_b in range(len(ebk)):
			#offsetting ebk so that for each band zero is the minimum energy of that band
			ebk = ebk - np.amin(ebk[i_b])
			val_mu = val_mu - np.amin(ebk[i_b])
			for i_k in range(len(ebk[0])):
				# se l'energia è nulla allora gli do un valore molto piccolo, vicino allo zero.
				if(ebk[i_b,i_k] == 0):
					#tau[i_T,i_b,i_k] = 0
					tau[i_T,i_b,i_k] = relaxTau.tau_tot(val_T, val_mu/units.Joule, 1e-6/units.Joule,rep,dop[i_T])
				else:
					tau[i_T,i_b,i_k] = relaxTau.tau_tot(val_T, val_mu/units.Joule, ebk[i_b,i_k]/units.Joule,rep,dop[i_T] )
		print('computed tau for T={}'.format(val_T))
	que.put(tau)
	
def tau_T_kp(Tr,kp,ebk,dop,que):
	"""
	compute tau(T,mu) for a given array of temperatures and chemical potentials. It is structured for multiprocessing with several sub-arrays of temperature working simultaneously. use atomic units.
	inputs: 
		- array of temperatures
		- array of chemical potential
		- e_bk
		- queue for multiprocessing
	outputs: 
		- tau_dft(T,mu)
	"""
	tau = np.zeros((len(Tr), len(kp), len(ebk), len(ebk[0])))
	for i_T,val_T in enumerate(Tr):
		for i_mu,val_mu in enumerate(kp):
			for i_b in range(len(ebk)):
				#offsetting ebk so that for each band zero is the minimum energy of that band
				ebk = ebk - np.amin(ebk[i_b])
				val_mu = val_mu - np.amin(ebk[i_b])
				for i_k in range(len(ebk[0])):
					# se l'energia è nulla allora gli do un valore molto piccolo, vicino allo zero.
					if(ebk[i_b,i_k] == 0):
						#tau[i_T,i_b,i_k] = 0 
						tau[i_T,i_mu,i_b,i_k] = relaxTau.tau_tot(val_T, val_mu/units.Joule, 1e-6/units.Joule,1,dop[i_T])
					else:
						tau[i_T,i_mu,i_b,i_k] = relaxTau.tau_tot(val_T, val_mu/units.Joule, ebk[i_b,i_k]/units.Joule,1,dop[i_T] )
		print('computed tau for T={}'.format(val_T))
	que.put(tau)


def fermIntegral_T(mu,e_bk,vxv_bk,npts,Tr,datatau,latVec,taubk,que,tmp_splt,i_tmp_splt):
	"""
	compute fermi integrals using the function from BoltzTraP2.bandlib. Structured for multiprocessing with several sub-arrays of temperature working simultaneously.
	Inputs:
	- array of chemical potentials
	- energy of bands e(b,#k)
	- vxv_bk(b,#k)
	- npts. number of points to subdivide energy in
	- sub-array of temperatures
	- index of particular temperature
	- vector of splitted temperatures
	- index of splitted temperatures sub-array
	"""
	# compute adder. it is used afterwards to get correct tau from tau_bk
	add = 0
	for i_tmp_sp,tmp_sp in enumerate(tmp_splt):
		if(i_tmp_sp < i_tmp_splt ):
			add = add + len(tmp_sp)
		else:
			break
	
	# defining shapes of N,L0,...,Lm11 vectors needed.
	N = np.zeros((len(Tr),len(mu))); L0 = np.zeros((len(Tr),len(mu),3,3)); L1 = np.zeros((len(Tr),len(mu),3,3)); 
	L2 = np.zeros((len(Tr),len(mu),3,3)); Lm11 = np.zeros((len(Tr),len(mu),3,3,3));
	
	
	
	for i_T,val_T in enumerate(Tr):
		ene, dos, vxvDos, curvDos = BL.BTPDOS(e_bk, vxv_bk, npts=npts, scattering_model=taubk[i_T+add])
		
		# save taubk and ebk in files only for one temperature
		#if (val_T == 300.0):
			#outputNameJSON = datadir_out + '/' + radixType + '_btp-NCRTM0-{}-{}-{}-{}-{}-{}.json'.format('taubk',mult,npts,nTemps,n_mu_pts,val_T)
			#with open(outputNameJSON,'w') as fil:
				#json.dump(taubk[i_T+add], fil, cls=NumpyEncoder)
			
			#outputNameJSON = datadir_out + '/' + radixType + '_btp-NCRTM0-{}-{}-{}-{}-{}-{}.json'.format('ene_int',mult,npts,nTemps,n_mu_pts,val_T)
			#with open(outputNameJSON,'w') as fil:
				#json.dump(ene, fil, cls=NumpyEncoder)
			
			#outputNameJSON = datadir_out + '/' + radixType + '_btp-NCRTM0-{}-{}-{}-{}-{}-{}.json'.format('dos_int',mult,npts,nTemps,n_mu_pts,val_T)
			#with open(outputNameJSON,'w') as fil:
				#json.dump(dos, fil, cls=NumpyEncoder)
			
			#outputNameJSON = datadir_out + '/' + radixType + '_btp-NCRTM0-{}-{}-{}-{}-{}.json'.format('ebk',mult,npts,nTemps,n_mu_pts)
			#with open(outputNameJSON,'w') as fil:
				#json.dump(e_bk, fil, cls=NumpyEncoder)
			#print('saved taubk {} and ebk {} for temperature {} K'.format(taubk[i_T+add].shape, e_bk.shape, val_T))
		
		x1,x2,x3,x4,x5 = BL.fermiintegrals(ene, dos, vxvDos, mur=mu, Tr=np.array([val_T])) # here we are using only one temperature, namely val_T. the for cycle on temperatures allows to compute coefficients N,L0,... for all T (and mu)
	
		N[i_T]=x1; L0[i_T]=x2; L1[i_T]=x3; L2[i_T]=x4; #Lm11[i_T]=x5[0] # here we are saving to ndarrays N,L0,...
	
		print('computed fermiIntegrals for T=',Tr[i_T])
	que.put([N,L0,L1,L2])
	

def dos_skim(ebk,vxvbk,npts):
	"""
	compute dos_tau from e_bk, vxvbk and npts
	input:
		- e_bk energy bands
		- vxv_bk tensor product of velocities
		- npts number of subdivisions of energy
	output:
		- dos density of states for each energy
 	"""
	ene_tau, dos_tau, vxvDos_tau, curvDos_tau = BL.BTPDOS(e_bk, vxv_bk, npts=npts)
	
	return ene_tau,dos_tau
