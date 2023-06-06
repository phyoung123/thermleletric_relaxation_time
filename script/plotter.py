#!/usr/bin/env python3
import ast
import itertools
import os.path
import codecs,json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import funzVar as fv
import ase.dft.kpoints as asekp
from BoltzTraP2.serialization import load_calculation as LC
from BoltzTraP2.misc import TimerContext
from BoltzTraP2.fite import getBands
datadir_in = os.path.abspath(os.path.dirname(__file__))
datadir_in = os.path.abspath(os.path.join(datadir_in, '../data/'))

# parameters of simulation we are going to plot:
inputs = {'multiplier':'nan','n_ene_points':'nan','nTemperatures':'nan','n_mu_points':'nan','radixType':'nan'}

greek_letterz=[chr(code) for code in range(945,970)]
greektau = greek_letterz[19]
har=27.21136
                                
with open('../input/input','r') as in_file:
	for line in in_file:
		words = line.split()
		for param in inputs:
			if len(words) != 0:
				if words[0] == param:
					for i_wrd,wrd in enumerate(words):
						if wrd == '=':
							inputs[param] = words[i_wrd+1]

# checking for errors in reading inputs
for i in inputs:
	if inputs[i] == 'nan':
		print('error! unable to load ',i,'. check input file for syntax. exiting')
		sys.exit()

print('INPUTS')
for i in inputs:
	print(i,inputs[i])
print()

npt = float(inputs['multiplier']) # multiplicative factor for k-points grid
n_ene = int(inputs['n_ene_points'])	# number of bins the energy ene is divided in bins. 
ntemp = int(inputs['nTemperatures'])	# number of energy to check
n_mu_pts = int(inputs['n_mu_points']) # number of divisions for chemical potential mu

radixType = inputs['radixType']

calclist=['ART','CRT','NCRT']

go_on=False
while not go_on:
	calctype = input('Enter calc [ART, NCRT (default), CRT) comma-sep : ').split(',')
	if calctype==['']:
		calctype=['NCRT']
		go_on=True
	for this in calctype:
		for that in calclist:
			if this==that:
				go_on=True

radix = calctype

print(f'\n========\nCurrent plotting choices:\n')

nstuff=20
stuffstatus=nstuff*[0]

stufftoplot  = ['S','sigma','ke','ZT','PF']
stufftoplot += ['Tr(S)','Tr(s)','Tr(ke)','Tr(ZT)','Tr(PF)']
stufftoplot += ['doping','DOS','tau_ave (ART)','tau_cumul (ART)']
stufftoplot += ['ZTmax(T)','tau(E,T)','Bands','Bands+n(E)','S,s,ke,PF vs doping','S,s,ke,PF vs T']

for ind,item in enumerate(stuffstatus):
        lab='no'
        if bool(item): lab='yes'
        print(f'{stufftoplot[ind]} (n.{ind}): {lab}')

l=input(f'\nEnter 1 to activate plots, anythg not to: ')
if l=='1':
	while True:
		l=input('Enter id to activate (comma-sep; "all" for all):').split(',')
		if l[0]=='':
			break
		elif l[0]=='all':
			stuffstatus=nstuff*[1]
			break
		else:
			choice = [int(x) for x in l]
			choiceok = all([i<nstuff for i in choice])
			choiceok = choiceok and all([i>=0 for i in choice])
			if choiceok:
				for ind,item in enumerate(choice):
					stuffstatus[item]=1
				break

l=input(f'\nEnter 1 to deactivate plots, anythg not to: ')
if l=='1':
	while True:
		l=input('Enter id to activate (comma-sep; "all" for all):').split(',')
		if l[0]=='':
			break
		elif l[0]=='all':
			stuffstatus=nstuff*[0]
			break
		else:
			choice = [int(x) for x in l]
			choiceok = all([i<nstuff for i in choice])
			choiceok = choiceok and all([i>=0 for i in choice])
			if choiceok:
				for ind,item in enumerate(choice):
					stuffstatus[item]=0
				break
	
                
print(f'\nNow plotting:')
for ind,item in enumerate(stuffstatus):
        if item==1: print(f'{stufftoplot[ind]}')
print()

c_seebeck= stuffstatus[0]
c_sigma  = stuffstatus[1]
c_ke = stuffstatus[2]
c_ZT = stuffstatus[3]
c_PF = stuffstatus[4]
c_seebeck_tr = stuffstatus[5]
c_sigma_tr = stuffstatus[6]
c_ke_tr = stuffstatus[7]
c_ZT_tr = stuffstatus[8]
c_Sxs_tr = stuffstatus[9]
c_doping = stuffstatus[10]
c_dos = stuffstatus[11]
c_tau_ave = stuffstatus[12]
c_tau_cum = stuffstatus[13]
c_max_ZT = stuffstatus[14]
c_tau_E_T = stuffstatus[15]
c_ebk=stuffstatus[16]
c_bn=stuffstatus[17]
c_alld=stuffstatus[18]
c_allT=stuffstatus[19]

if c_ebk or c_dos or c_bn:
	bt2_file=f'{radixType}_btp{npt}.bt2'
	print(f'Loading interpolation from {bt2_file}...',flush=True)
	data, equivalences, coeffs, metadata = LC(bt2_file)
	print('Done')
	print(data.atoms.cell,data.atoms.get_volume())

radix2 = []	# this is for relaxation time average plots only for ART
radix3 = [] # this is for skimmed energies and skimmed dos. only for NCRT or NCRTM0
radix4 = [] # same as radix 2 but for ARTM0,NCRT,NCRTM0

        
# carica i dati. con np.asarray li de-jsonizza e le rende numpy.ndarray-----
for rd in radix:
	param = ['info',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	info = json.load(open(inputNameJSON))
	info = np.asarray(info)
	print('\nLoading interpolations\nNow plotting ',rd,'.')

# -------------------------------------------------------------------------

Temperatures={}; chemPot={}; Nelect={}; doping={}; energie={}
optimal_mu={}; dos={}; dos_skim={}; ene_skim={}; kpts={}; dims={}; 
# Get Temperature and chemical potential and doping and energies array -----------------
for rd in radix:
	param = ['Temp',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	Temperatures[rd] = json.load(open(inputNameJSON))
	Temperatures[rd] = np.asarray(Temperatures[rd])
	nTemp = len(Temperatures[rd])
        
	param = ['chemPot',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	chemPot[rd] = json.load(open(inputNameJSON))
	chemPot[rd] = np.asarray(chemPot[rd])

	param = ['Nelect',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	Nelect[rd] = json.load(open(inputNameJSON))
	Nelect[rd] = np.asarray(Nelect[rd])

	param = ['doping',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	doping[rd] = json.load(open(inputNameJSON))
	doping[rd] = np.asarray(doping[rd])

	param = ['energie',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	energie[rd] = json.load(open(inputNameJSON))
	energie[rd] = np.asarray(energie[rd])
	
	param = ['optimal_mu',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	optimal_mu[rd] = json.load(open(inputNameJSON))
	optimal_mu[rd] = np.asarray(optimal_mu[rd])
	
	param = ['dos',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	dos[rd] = json.load(open(inputNameJSON))
	dos[rd] = np.asarray(dos[rd])
	
	param = ['kpoints',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	kpts[rd] = json.load(open(inputNameJSON))
	kpts[rd] = np.asarray(kpts[rd])
	
	param = ['dims',rd]
	inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
	dims[rd] = json.load(open(inputNameJSON))
	dims[rd] = np.asarray(dims[rd])
	

#### variosu graph related stuff
# plus thickness of lines in T-doping plots

plt.rcParams.update({'font.size': 18})
fontlab = {'family' : 'serif',
	       'size'   : 18}
plt.rc('font', **fontlab)
plt.rcParams["figure.figsize"] = [9.5,7]
ticksize=18
labsize=20
legsize=14
#plt.rc('text', usetex=True)

thk0=0.6;thkm=6;thkd=(thkm/thk0)**(1/nTemp)

# plot seebeck -----

if c_seebeck:
	seebeck = {}
	for rd in radix:
		param = ['seebeck',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		seebeck[rd] = json.load(open(inputNameJSON))
		seebeck[rd] = np.asarray(seebeck[rd])
		icompl=['a','b','c']
		for icomp in range(3):
			fig,ax1 = plt.subplots()
			fig.suptitle(f'S, comp.{icomp}, '+rd)
			ax1.set_xscale('log')
			#ax1.set_yscale('log')
			ax1.set_xlabel('Carrier density (cm$^{-3}$')
			ax1.set_ylabel('S ($\mu$V/K) '+rd)
			thk=thk0
			caz=[]
			caz1=[]
			for i_T,val_T in enumerate(Temperatures[rd]):
				if True:#i_T%20==0 and val_T>100:
					ax1.plot(-doping[rd][i_T] , abs(1e6*seebeck[rd][i_T,:,icomp,icomp]),lw=thk,label='T='+str(val_T))
					ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], abs(1e6*seebeck[rd][i_T, optimal_mu[rd][i_T]][icomp,icomp]) ,'s')		
					caz.append(-doping[rd][i_T][optimal_mu[rd][i_T]])
					caz1.append(abs(1e6*seebeck[rd][i_T,optimal_mu[rd][i_T],icomp,icomp]))
					thk*=thkd
			ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
			ax1.legend()
			plt.figure(990)
			plt.title(f'S vs T '+rd)
			plt.xlabel('Temperature (K)')
			plt.ylabel('S ($\mu$V/K) '+rd)
			plt.plot(Temperatures[rd],np.asarray(caz1),'-',label=icompl[icomp])
			plt.legend()
		plt.show()

# plot sigma ---------------------------------------------------------------------
if c_sigma:
	sigma = {}
	for rd in radix:
		param = ['sigma',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		sigma[rd] = json.load(open(inputNameJSON))
		sigma[rd] = np.asarray(sigma[rd])

		icompl=['a','b','c']
		for icomp in range(3):
			fig,ax1 = plt.subplots()
			fig.suptitle(f'sigma, comp.{icomp}, '+rd)
			ax1.set_xscale('log')
			#ax1.set_yscale('log')
			ax1.set_xlabel('Carrier density (cm$^{-3}$')
			ax1.set_ylabel('$\sigma$ (S m) '+rd)
			thk=thk0
			caz=[]
			caz1=[]
			for i_T,val_T in enumerate(Temperatures[rd]):
				if True:#i_T%20==0 and val_T>100:
					ax1.plot(-doping[rd][i_T] , 1e-3*sigma[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
					ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], 1e-3*sigma[rd][i_T, optimal_mu[rd][i_T]][icomp,icomp],'s')
					caz.append(-doping[rd][i_T][optimal_mu[rd][i_T]])
					caz1.append(1e-3*sigma[rd][i_T,optimal_mu[rd][i_T],icomp,icomp])
					thk*=thkd
			ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
			plt.figure(991)
			plt.title(f'sigma vs T '+rd)
			plt.xlabel('Temperature (K)')
			plt.ylabel('$\sigma$ (kS/m) '+rd)
			plt.plot(Temperatures[rd],np.asarray(caz1),'-',label=icompl[icomp])
			ax1.legend()
		plt.show()

# ------------------------------------------------------------------------
if c_alld or c_allT:

	ke = {}; s = {}; S = {}; PF = {}
	print(f'\nPlotting subplots vs T or doping.\nBEWARE: axis parameters,labels etc must be customized.')
	for rd in radix:		
		param = ['ke',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ke[rd] = json.load(open(inputNameJSON))
		ke[rd] = np.asarray(ke[rd])	
		param = ['seebeck',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		S[rd] = json.load(open(inputNameJSON))
		S[rd] = abs(np.asarray(S[rd]))
		param = ['sigma',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		s[rd] = json.load(open(inputNameJSON))
		s[rd] = np.asarray(s[rd])
		PF[rd]= 1000*s[rd]*S[rd]*S[rd]
		S[rd] *= 1e6
		s[rd] *= 1e-6
		print(radix,rd,type(radix),type(rd))

		if c_alld:
			fig,ax = plt.subplots(2,2)	
			fig.subplots_adjust(wspace=0.01,hspace=0.01)
			ax1=ax[0][0]
			ax2=ax[0][1]
			ax3=ax[1][0]
			ax4=ax[1][1]
			ax1.tick_params(right= False ,top= False,left= True, bottom= True)
			ax2.tick_params(right= True,top= False,left= False, bottom= True)
			ax3.tick_params(right= False ,top= False,left= True, bottom= True)
			ax4.tick_params(right=True,top= False,left= False, bottom= True)
				
			axlist=[ax1,ax2,ax3,ax4]
			for axx in axlist:
				axx.set_xscale('log')
				axx.set_xlim(0.01,1)
			ax1.set_ylim(0.001,600)
			ax2.set_ylim(0.001,1)
			ax3.set_ylim(0.001,3)
			ax4.set_ylim(0.001,23)

			ax3.set_xlabel('Carrier density (10$^{21}$ cm$^{-3}$)')
			ax4.set_xlabel('Carrier density (10$^{21}$ cm$^{-3}$)')
			ax2.yaxis.set_label_position("right")
			ax4.yaxis.set_label_position("right")
			ax2.yaxis.tick_right()
			ax4.yaxis.tick_right()
			ax1.set_ylabel('|S| ($\mu$V/k)') 
			ax2.set_ylabel('$\sigma$ (MS/m)',rotation=270,ha='center',va='baseline',rotation_mode='anchor')
			ax3.set_ylabel('$k_e$ (W/K/m)')
			ax4.set_ylabel('Power factor (mW/(K$^2$ m)',rotation=270,ha='center',va='baseline',rotation_mode='anchor')
			ax1.set_xticklabels([])
			ax2.set_xticklabels([]) 

			thk=thk0
			caz=[]; caz1=[]; caz2=[]; caz3=[]; caz4=[]
			icomp=1; sca=1e-21   #b-axis, density in units of 10^21 
			for i_T,val_T in enumerate(Temperatures[rd]):
				if (i_T+1)%10==0 and val_T>100:
					ax1.plot(-sca*doping[rd][i_T] , S[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
					ax2.plot(-sca*doping[rd][i_T] , s[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
					ax3.plot(-sca*doping[rd][i_T] , ke[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
					ax4.plot(-sca*doping[rd][i_T] , PF[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
					caz.append(-sca*doping[rd][i_T][optimal_mu[rd][i_T]])
					caz1.append(S[rd][i_T,optimal_mu[rd][i_T],1,1])
					caz2.append(s[rd][i_T,optimal_mu[rd][i_T],1,1])
					caz3.append(ke[rd][i_T,optimal_mu[rd][i_T],1,1])
					caz4.append(PF[rd][i_T,optimal_mu[rd][i_T],1,1])
					thk*=thkd*1.2
			ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
			ax2.plot(caz,caz2,marker='s',linestyle='dashed',color='red')
			ax3.plot(caz,caz3,marker='s',linestyle='dashed',color='red')
			ax4.plot(caz,caz4,marker='s',linestyle='dashed',color='red')
			plt.tight_layout(0.1)
			plt.show()

		if c_allT:
			figx,axi = plt.subplots(2,2)
			figx.subplots_adjust(wspace=0.01,hspace=0.01)
			ax1=axi[0][0]
			ax2=axi[0][1]
			ax3=axi[1][0]
			ax4=axi[1][1]
			ax2.yaxis.set_label_position("right")
			ax4.yaxis.set_label_position("right")
			ax3.set_xlabel('Temperature (K)')
			ax4.set_xlabel('Temperature (K)')
			ax2.yaxis.set_label_position("right")
			ax4.yaxis.set_label_position("right")
			ax2.yaxis.tick_right()
			ax4.yaxis.tick_right()
			ax1.set_ylabel('|S| ($\mu$V/k)')
			ax2.set_ylabel('$\sigma$ (MS/m)',rotation=270,ha='center',va='baseline', rotation_mode='anchor')
			ax3.set_ylabel('$k_e$ (W/K/m)')
			ax4.set_ylabel('Power factor  (mW/(K$^2$ m)',rotation=270,ha='center',va='baseline', rotation_mode='anchor')
			
			ax1.set_xticklabels([]) 
			ax2.set_xticklabels([]) 
			axlists=[ax1,ax2,ax3,ax4]
			for axx in axlists:
				axx.set_xlim(200,1100)
			ax4.set_xlim(200.01,1100)
			ax1.set_ylim(0.001,400)
			ax2.set_ylim(0.001,0.7)
			ax3.set_ylim(0.001,3)
			ax4.set_ylim(0.001,17)
			NN=7
			print(f'\nSubplt vs T: plotting every {NN} temperatures\n')
			Sx=[S[rd][iT][optimal_mu[rd][iT]][0,0] for iT in range(len(Temperatures[rd])) if iT%NN==0]
			Sy=[S[rd][iT][optimal_mu[rd][iT]][1,1] for iT in range(len(Temperatures[rd])) if iT%NN==0]
			Sz=[S[rd][iT][optimal_mu[rd][iT]][2,2] for iT in range(len(Temperatures[rd])) if iT%NN==0]
			tt = [Temperatures[rd][iT] for iT in range(len(Temperatures[rd]))if iT%NN==0]

			sx=[s[rd][iT][optimal_mu[rd][iT]][0,0] for iT in range(len(Temperatures[rd])) if iT%NN==0]
			sy=[s[rd][iT][optimal_mu[rd][iT]][1,1] for iT in range(len(Temperatures[rd])) if iT%NN==0] 
			sz=[s[rd][iT][optimal_mu[rd][iT]][2,2] for iT in range(len(Temperatures[rd])) if iT%NN==0] 

			kex=[ke[rd][iT][optimal_mu[rd][iT]][0,0] for iT in range(len(Temperatures[rd])) if iT%NN==0]
			key=[ke[rd][iT][optimal_mu[rd][iT]][1,1] for iT in range(len(Temperatures[rd])) if iT%NN==0] 
			kez=[ke[rd][iT][optimal_mu[rd][iT]][2,2] for iT in range(len(Temperatures[rd])) if iT%NN==0] 
			PFx=[PF[rd][iT][optimal_mu[rd][iT]][0,0] for iT in range(len(Temperatures[rd])) if iT%NN==0]
			PFy=[PF[rd][iT][optimal_mu[rd][iT]][1,1] for iT in range(len(Temperatures[rd])) if iT%NN==0] 
			PFz=[PF[rd][iT][optimal_mu[rd][iT]][2,2] for iT in range(len(Temperatures[rd])) if iT%NN==0] 

			ax1.tick_params(right= False ,top= False,left= True, bottom= True)
			ax2.tick_params(right= True,top= False,left= False, bottom= True)
			ax3.tick_params(right= False ,top= False,left= True, bottom= True)
			ax4.tick_params(right=True,top= False,left= False, bottom= True)
			#	ax2.set_xlabel('T (K)')

			ax1.plot(tt,Sx,'-',label='a')
			ax1.plot(tt,Sy,'.-',label='b')
			ax1.plot(tt,Sz,'--',label='c')
			ax2.plot(tt,sx,'-',label='a')
			ax2.plot(tt,sy,'.-',label='b')
			ax2.plot(tt,sz,'--',label='c')
			ax3.plot(tt,kex,'-',label='a')
			ax3.plot(tt,key,'.-',label='b')
			ax3.plot(tt,kez,'--',label='c')
			ax4.plot(tt,PFx,'-',label='a')
			ax4.plot(tt,PFy,'.-',label='b')
			ax4.plot(tt,PFz,'--',label='c')
			for axx in axlists:
				axx.legend()
			plt.tight_layout(0.1)
			plt.show()


# plot ke  ---------------------------------------------------------------------
if c_ke:
	ke = {}
	for rd in radix:
		param = ['ke',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ke[rd] = json.load(open(inputNameJSON))
		ke[rd] = np.asarray(ke[rd])

		icompl=['a','b','c']
		for icomp in range(3):
			fig,ax1 = plt.subplots()
			fig.suptitle(f'k_e, comp.{icomp}, '+rd)
			ax1.set_xscale('log')
			#ax1.set_yscale('log')
			ax1.set_xlabel('Carrier density (cm$^{-3}$')
			ax1.set_ylabel('k$_e$ (W/K/m) '+rd)
			thk=thk0
			caz=[]
			caz1=[]
			for i_T,val_T in enumerate(Temperatures[rd]):
				if True:#i_T%20==0 and val_T>100:
					ax1.plot(-doping[rd][i_T],ke[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
					ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], ke[rd][i_T, optimal_mu[rd][i_T]][icomp,icomp] ,'s')		
					caz.append(-doping[rd][i_T][optimal_mu[rd][i_T]])
					caz1.append(ke[rd][i_T,optimal_mu[rd][i_T],icomp,icomp])
					thk*=thkd
			ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
			plt.figure(991)
			plt.title(f'k_e vs T '+rd)
			plt.xlabel('Temperature (K)')
			plt.ylabel('k_e (W/K/m) '+rd)
			plt.plot(Temperatures[rd],np.asarray(caz1),'-',label=icompl[icomp])
			ax1.legend()
		plt.show()

# plot ZT  ---------------------------------------------------------------------
if c_ZT:
	ZT = {}
	for rd in radix:
		param = ['ZT',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ZT[rd] = json.load(open(inputNameJSON))
		ZT[rd] = np.asarray(ZT[rd])
		
		fig,ax1 = plt.subplots()
		icompl=['a','b','c']
		for icomp in range(3):
			fig.suptitle(f'ZT, comp.{icomp}, '+rd)
			ax1.set_xscale('log')
			#ax1.set_yscale('log')
			ax1.set_xlabel('Carrier density (cm$^{-3}$')
			ax1.set_ylabel('$\sigma$S$^2$ ($SV^2/m K^2$) '+rd)
			thk=thk0
			caz1=[];caz=[]
			for i_T,val_T in enumerate(Temperatures[rd]):
				if True:#i_T%20==0 and val_T>100:
					ax1.plot(-doping[rd][i_T] ,abs(ZT[rd][i_T,:,icomp,icomp]),lw=thk,label='T='+str(val_T))
					caz.append(-doping[rd][i_T][optimal_mu[rd][i_T]])
					caz1.append(ZT[rd][i_T,optimal_mu[rd][i_T],icomp,icomp])
					thk*=thkd
			ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
			ax1.legend()
		plt.show()

	ZT = {}
	for rd in radix:
		param = ['ZT',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ZT[rd] = json.load(open(inputNameJSON))
		ZT[rd] = np.asarray(ZT[rd])
		
		fig,(ax2,ax1) = plt.subplots(1,2)
		fig.subplots_adjust(wspace=0.01,hspace=0.01)
		ax1.tick_params(right= True ,top= False,left= True, bottom= True)
		ax2.tick_params(right= True ,top= False,left= True, bottom= True)

	#	fig.suptitle(f'ZT, comp.{icomp}, '+rd)
		ax1.set_xscale('log')
		ax1.set_xlabel('Carrier density (cm$^{-3}$)')
		ax1.yaxis.set_label_position("right")
		ax1.yaxis.tick_right()
		ax1.set_ylabel('ZT vs doping and T',rotation=270,ha='center',va='baseline', rotation_mode='anchor')
		thk=thk0
		caz=[]
		caz1=[]
		ax1.set_ylim(0,7)
		ax1.set_xlim(1e19,5e21)
		icomp=1
		with open('dummy.dat','w') as f:
			f.write(str(ZT[rd]))
		for i_T,val_T in enumerate(Temperatures[rd]):
			if (i_T+1)%10==0 and val_T>100:
				ax1.plot(-doping[rd][i_T] , ZT[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(val_T))
				caz.append(-doping[rd][i_T][optimal_mu[rd][i_T]])
				caz1.append(ZT[rd][i_T,optimal_mu[rd][i_T],icomp,icomp])
				thk*=thkd*1.2
		ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
		#ax1.legend()
		ax2.set_ylim(0,7)
		ax2.set_xlim(100,1101)
		ax2.set_xlabel('Temperature (K)')
		ax2.set_ylabel('ZT at optimal doping')

		ZTx=[ZT[rd][iT][optimal_mu[rd][iT]][0,0] for iT in range(len(Temperatures[rd]))]
		ZTy=[ZT[rd][iT][optimal_mu[rd][iT]][1,1] for iT in range(len(Temperatures[rd]))] 
		ZTz=[ZT[rd][iT][optimal_mu[rd][iT]][2,2] for iT in range(len(Temperatures[rd]))] 
		
		ax2.set_xlabel('T (K)')
		ax2.plot(Temperatures[rd],ZTx,'-',label='a')
		ax2.plot(Temperatures[rd],ZTy,'.-',label='b')
		ax2.plot(Temperatures[rd],ZTz,'.',label='c')
		ax2.legend()
		plt.tight_layout(0.1)
		plt.show()

# plot PF ---------------------------------------------------------------------

if c_PF:
	Sy = {}
	sy = {}
	for rd in radix:
		param = ['seebeck',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		Sy[rd] = json.load(open(inputNameJSON))
		Sy[rd] = np.asarray(Sy[rd])
		
		param = ['sigma',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		sy[rd] = json.load(open(inputNameJSON))
		sy[rd] = np.asarray(sy[rd])

		icompl=['a','b','c']
		for icomp in range(3):
			fig,ax1 = plt.subplots()
			fig.suptitle(f'Power factor, comp.{icomp}, '+rd)
			ax1.set_xscale('log')
			#ax1.set_yscale('log')
			ax1.set_xlabel('Carrier density (cm$^{-3}$')
			ax1.set_ylabel('$\sigma$S$^2$ ($SV^2/m K^2$) '+rd)
			thk=thk0
			caz=[]
			caz1=[]
			for i_T,val_T in enumerate(Temperatures[rd]):
				if True:#i_T%20==0 and val_T>100:
					ax1.plot(-doping[rd][i_T], Sy[rd][i_T,:,icomp,icomp]**2*sy[rd][i_T,:,icomp,icomp],lw=thk,label='T='+str(int(val_T)))
					ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], Sy[rd][i_T,optimal_mu[rd][i_T],icomp,icomp]**2*sy[rd][i_T,optimal_mu[rd][i_T],icomp,icomp],marker='s',linestyle='dashed')
					caz.append(-doping[rd][i_T][optimal_mu[rd][i_T]])
					caz1.append(Sy[rd][i_T,optimal_mu[rd][i_T],icomp,icomp]**2*sy[rd][i_T,optimal_mu[rd][i_T],icomp,icomp])
					thk*=thkd
			ax1.plot(caz,caz1,marker='s',linestyle='dashed',color='red')
			plt.figure(993)
			plt.title(f'Power factor vs T - {rd}')
			plt.xlabel('Temperature (K)')
			plt.ylabel('Power factor ($m$$S$$V^2$/m $K^2$) '+rd)
			plt.plot(Temperatures[rd],1000*np.asarray(caz1),'-',label=icompl[icomp])
			ax1.legend()
		plt.show()


# ------------------------------------------------------------------------

# plot sigma tr ---------------------------------------------------------------------
if c_sigma_tr:
	sigma_tr = {}
	for rd in radix:
		param = ['sigma_tr',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		sigma_tr[rd] = json.load(open(inputNameJSON))
		sigma_tr[rd] = np.asarray(sigma_tr[rd])

		fig,ax1 = plt.subplots()
		fig.suptitle('sigma tr '+rd)
		ax1.set_xscale('log')
		#ax1.set_yscale('log')
		ax1.set_xlabel('doping cm^-3')
		ax1.set_ylabel('sigma tr (kS/m) '+rd)

		for i_T,val_T in enumerate(Temperatures[rd]):
			ax1.plot(-doping[rd][i_T] , sigma_tr[rd][i_T]*1e-3 ,label='T='+str(val_T))
			ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], abs(sigma_tr[rd][i_T, optimal_mu[rd][i_T]]*1e-3) ,'*')
		ax1.legend()
# ------------------------------------------------------------------------

# plot ke tr ---------------------------------------------------------------------
if c_ke_tr:
	ke_tr = {}
	for rd in radix:
		param = ['ke_tr',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ke_tr[rd] = json.load(open(inputNameJSON))
		ke_tr[rd] = np.asarray(ke_tr[rd])

		fig,ax1 = plt.subplots()
		fig.suptitle('ke tr '+rd)
		ax1.set_xscale('log')
		ax1.set_xlabel('doping cm^-3')
		ax1.set_ylabel('ke tr '+rd)

		for i_T,val_T in enumerate(Temperatures[rd]):
			ax1.plot(-doping[rd][i_T] , ke_tr[rd][i_T] ,label='T='+str(val_T))
			ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], abs(ke_tr[rd][i_T, optimal_mu[rd][i_T]]) ,'*')
		ax1.legend()
# ------------------------------------------------------------------------

# plot S tr ---------------------------------------------------------------------
if c_seebeck_tr:
	seebeck_tr = {}
	for rd in radix:
		param = ['seebeck_tr',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		seebeck_tr[rd] = json.load(open(inputNameJSON))
		seebeck_tr[rd] = np.asarray(seebeck_tr[rd])

		fig,ax1 = plt.subplots()
		fig.suptitle('Seebeck tr '+rd)
		ax1.set_xscale('log')
		ax1.set_xlabel('doping cm^-3')
		ax1.set_ylabel('seebeck tr ($\mu V/K$) '+rd)

		for i_T,val_T in enumerate(Temperatures[rd]):
			ax1.plot(-doping[rd][i_T] , abs(seebeck_tr[rd][i_T]*1e6 ),label='T='+str(val_T))
			ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], abs(seebeck_tr[rd][i_T, optimal_mu[rd][i_T]]*1e6) ,'*')
		ax1.legend()
# ------------------------------------------------------------------------

# plot ZT tr ---------------------------------------------------------------------
if c_ZT_tr:
	ZT_tr = {}; S={}; s={};ke={};ZT={}
	for rd in radix:
		param = ['seebeck',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		S[rd] = json.load(open(inputNameJSON))
		S[rd] = np.asarray(S[rd])

		param = ['ZT',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ZT[rd] = json.load(open(inputNameJSON))
		ZT[rd] = np.asarray(ZT[rd])

		param = ['ZT_tr',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ZT_tr[rd] = json.load(open(inputNameJSON))
		ZT_tr[rd] = np.asarray(ZT_tr[rd])
		param = ['ke',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ke[rd] = json.load(open(inputNameJSON))
		ke[rd] = np.asarray(ke[rd])	
		param = ['sigma',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		s[rd] = json.load(open(inputNameJSON))
		s[rd] = np.asarray(s[rd])

		kl=np.zeros(len(Temperatures[rd]))
		with open('../input/latthcond','r') as f:
			for iT,vT in enumerate(Temperatures[rd]):
				kl[iT] = float(f.readline())

		sTT=np.zeros_like(kl)
		kTT=np.zeros_like(kl)
		STT=np.zeros_like(kl)
		ZT0=np.zeros_like(kl)
		UZT=np.zeros_like(kl)

		for iT, vT in enumerate(Temperatures[rd]):

			dum1=0.; dum2=0.; dum3=0.
			for i in range(3):
				dum1 += 1./s[rd][iT,optimal_mu[rd][iT],i,i]
				dum2 += 1./ke[rd][iT,optimal_mu[rd][iT],i,i]
				dum3 += 1./ZT[rd][iT,optimal_mu[rd][iT],i,i]
				STT[iT]+=S[rd][iT,optimal_mu[rd][iT],i,i]
				ZT0[iT]+=ZT[rd][iT,optimal_mu[rd][iT],i,i]
			sTT[iT]=3./dum1
			kTT[iT]=3./dum2
			UZT[iT]=3./dum3
			
			STT[iT]/=3.;ZT0[iT]/=3.

		ZTT = sTT *STT**2/(kTT+kl)*Temperatures[rd]

		fig,ax1 = plt.subplots()
		#fig.suptitle('ZT tr '+rd)
#		ax1.set_xscale('log')
		ax1.set_xlabel('T (K)')
		ax1.set_ylabel('ZT')
		ax1.set_ylim(0,1)
		ax1.plot(Temperatures[rd],ZTT,'.-',label='ZT with harmonic ave of $\sigma$, $k_e$')
		ax1.plot(Temperatures[rd],UZT,'-v',label='Harmonic average of ZT')
		ax1.plot(Temperatures[rd],ZT0,'--',label='Trace of ZT')

#		for i_T,val_T in enumerate(Temperatures[rd]):
#			ax1.plot(-doping[rd][i_T] , ZT_tr[rd][i_T] ,label='T='+str(val_T))
#			ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], abs(ZT_tr[rd][i_T, optimal_mu[rd][i_T]]) ,'*')
		ax1.legend()
# ------------------------------------------------------------------------

if c_Sxs_tr:
	S_tr = {}
	s_tr = {}
	for rd in radix:
		param = ['seebeck_tr',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		S_tr[rd] = json.load(open(inputNameJSON))
		S_tr[rd] = np.asarray(S_tr[rd])
		
		param = ['sigma_tr',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		s_tr[rd] = json.load(open(inputNameJSON))
		s_tr[rd] = np.asarray(s_tr[rd])

		fig,ax1 = plt.subplots()
		fig.suptitle('Seebeck$^2$ x sigma tr '+rd)
		ax1.set_xscale('log')
		#ax1.set_yscale('log')
		ax1.set_xlabel('doping cm^-3')
		ax1.set_ylabel('S$^2$ s ($SV^2/m K^2$) tr '+rd)

		for i_T,val_T in enumerate(Temperatures[rd]):
			ax1.plot(-doping[rd][i_T] , S_tr[rd][i_T]**2 * s_tr[rd][i_T] ,label='T='+str(val_T))
			ax1.plot(-doping[rd][i_T][optimal_mu[rd][i_T]], abs(S_tr[rd][i_T, optimal_mu[rd][i_T]]**2 * s_tr[rd][i_T, optimal_mu[rd][i_T]]) ,'*')
		ax1.legend()

# plot doping(mu) -----------------------------------------------------
if c_doping:
	for rd in radix:
		fig, ax = plt.subplots()
		fig.suptitle('doping '+rd+' (mu)')
		plt.xlabel('$\mu$ (eV)')
		plt.ylabel('Doping ($cm^{-3}$) '+rd)

		for i_T,val_T in enumerate(Temperatures[rd]):
			ax.plot(har*chemPot[rd] , doping[rd][i_T],label='T='+str(val_T))
#		ax.legend()

## plot dos(E) -----------------------------------------------------
if c_dos:
	volume=data.atoms.get_volume()*1e-24
	for rd in radix:
		fig, ax = plt.subplots()
		fig.suptitle('Density of states dos(E) '+rd)
		plt.xlabel('E (eV)')
		plt.ylabel('DOS (1/eV/cm$^3$)')
		ax.plot(har*energie[rd] , dos[rd]/har/volume)

# plot tau_ave(T,mu) ----------------------------------------------------
if c_tau_ave:

	if 'ART' in radix:
		rd = 'ART'
		param = ['tau_ave',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		tau_ave = json.load(open(inputNameJSON))
		tau_ave = np.asarray(tau_ave)


		fig, ax1 = plt.subplots()
		fig.suptitle(f'{greektau} (T,$\mu$) '+rd)
		ax1.set_xlabel('$\mu$ (Ha)')
		ax1.set_ylabel(f'{greektau} (s)')

		for i_T,val_T in enumerate(Temperatures[rd]):
			ax1.plot(chemPot[rd] , tau_ave[i_T],label='T='+str(val_T))
		ax1.legend()


		fig2, ax2 = plt.subplots()
		fig.suptitle(f'{greektau} (T) '+rd)
		ax2.set_xlabel('T (K)')
		ax2.set_ylabel(f'{greektau} (s)')
		
		id_mu = np.argmin(abs(chemPot[rd]-0))

		ax2.plot(Temperatures[rd] , tau_ave[:,id_mu])
		ax2.legend()
	else:
		print('This plot tau_ave is avaliable only for ART')

# plot tau_cum -----------------------------------------------------------------------
if c_tau_cum:
	if 'ART' in radix:
		rd = 'ART'
		param = ['tau_cum',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		tau_cum = json.load(open(inputNameJSON))
		tau_cum = np.asarray(tau_cum)

		param = ['ene_pos',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
		ene_pos = json.load(open(inputNameJSON))
		ene_pos = np.asarray(ene_pos)
		
		id_mu = np.argmin(abs(chemPot[rd]-0))
		
		fig,ax = plt.subplots()
		fig.suptitle(f'ART cumulative {greektau} for $\mu=${0:.2f}'.format(chemPot[rd][id_mu]))
		
		ax.set_xlabel('energy (Ha)')
		ax.set_ylabel(f'cumulative {greektau} (s)')
		for i_T,val_T in enumerate(Temperatures[rd]):
			ax.plot(ene_pos , tau_cum[i_T][id_mu] ,label='T='+str(val_T))
		ax.legend()

	else:
		print('This plot tau_cum is avaliable only for ART')

# plot max ZT(T)  ---------------------------------------------------------------------
if c_max_ZT:
	ZT = {}
	ZTx={}; ZTy={}; ZTz={};
	for ird,rd in enumerate(radix):
                
		param = ['ZT',rd]
		inputNameJSON = datadir_in + '/' +radixType+ '_btp-{}-{}-{}-{}-{}-{}.json'.format(param[1],param[0],npt,n_ene,ntemp,n_mu_pts)
                
		ZT[rd] = json.load(open(inputNameJSON))
		ZT[rd] = np.asarray(ZT[rd])
		ZTx[rd]=[ZT[rd][iT][optimal_mu[rd][iT]][0,0] for iT in range(len(Temperatures[rd]))]
		ZTy[rd]=[ZT[rd][iT][optimal_mu[rd][iT]][1,1] for iT in range(len(Temperatures[rd]))] 
		ZTz[rd]=[ZT[rd][iT][optimal_mu[rd][iT]][2,2] for iT in range(len(Temperatures[rd]))] 
		
		plt.rcParams.update({'font.size': 18})
		fig, axarr = plt.subplots()
		fig.set_figheight(6)
		fig.set_figwidth(8)
		axarr.set_xlabel('T (K)')
		axarr.set_ylabel('Max $ZT$ '+rd)
			
		axarr.plot(Temperatures[rd],ZTx[rd],'-',label='a')
		axarr.plot(Temperatures[rd],ZTy[rd],'.-',label='b')
		axarr.plot(Temperatures[rd],ZTz[rd],'.',label='c')
			
		axarr.legend()

# plot c_tau_E_component -------------------------------------------------------
if c_tau_E_T:
	
	Ens = []
	param = ['Ens']
	inputNameJSON = datadir_in + '/{}.json'.format(param[0])
	Ens = json.load(open(inputNameJSON))
	Ens = np.asarray(Ens)
	
	tmps = []
	param = ['Tmps']
	inputNameJSON = datadir_in + '/{}.json'.format(param[0])
	tmps = json.load(open(inputNameJSON))
	tmps = np.asarray(tmps)
	
	tau_ac = []
	param = ['tau_ac']
	inputNameJSON = datadir_in + '/{}.json'.format(param[0])
	tau_ac = json.load(open(inputNameJSON))
	tau_ac = np.asarray(tau_ac)
	
	tau_imp = []
	param = ['tau_imp']
	inputNameJSON = datadir_in + '/{}.json'.format(param[0])
	tau_imp = json.load(open(inputNameJSON))
	tau_imp = np.asarray(tau_imp)
	
	tau_pol = []
	param = ['tau_pol']
	inputNameJSON = datadir_in + '/{}.json'.format(param[0])
	tau_pol = json.load(open(inputNameJSON))
	tau_pol = np.asarray(tau_pol)
	
	tau_tot = []
	param = ['tau_tot']
	inputNameJSON = datadir_in + '/{}.json'.format(param[0])
	tau_tot = json.load(open(inputNameJSON))
	tau_tot = np.asarray(tau_tot)
	
	tuttitau = [tau_ac, tau_imp, tau_pol, tau_tot]
	taunomi = [greektau+'$_{ac}$', greektau+'$_{imp}$', greektau+'$_{pol}$', greektau+'$_{tot}$']
	
	iTarray = [int(len(tmps)/2)]
	Earray = [int(len(Ens)/2)]
	
	for ind,val in enumerate(iTarray):
		
		itemp = val
		
		fig, ax = plt.subplots()
		#fig.suptitle(f'{greektau}(E), T={round(tmps[itemp],0)} K, $\mu$=0')
		
		ax.set_yscale('log')
		ax.set_xlabel('E (eV)')
		ax.set_ylabel(f'{greektau} (fs) ')
		
		for iTau,vTau in enumerate(tuttitau):
			ax.plot(Ens  , 1e15*vTau[itemp],label=taunomi[iTau])
	ax.legend()

	for ind,val in enumerate(Earray):
		
		iEne = val
		
		fig, ax = plt.subplots()
		#fig.suptitle(f'{greektau}(T), E={round(1e3*Ens[iEne],0)} meV, $\mu$=0')
		
		ax.set_yscale('log')
		ax.set_xlabel('T (K))')
		ax.set_ylabel(greektau+' (fs) ')
		#ax.set_xlim(0,0.8)
		
		for iTau,vTau in enumerate(tuttitau):
			ax.plot(tmps  , 1e15*vTau[:,iEne],label=taunomi[iTau])
			
	ax.legend()


############## - end band plot
if c_bn:

	print('\nPlotting bands+n(E) (method independent)\n')

	plt.rcParams.update({'font.size': 18})
	fontlab = {'family' : 'serif',
        'size'   : 18}
	plt.rc('font', **fontlab)
	plt.rcParams["figure.figsize"] = [9.5,7]

	ticksize=18
	labsize=20
	legsize=14
	
	fig, axx = plt.subplots(1,3,gridspec_kw={'width_ratios': [2,2,1.5]})	
	fig.subplots_adjust(wspace=0.01,hspace=0.01)
	#
	ax=axx[0]
	ax3=axx[1]
	ax2=axx[2]

	volume = data.atoms.get_volume()*1e-24
	kbz = 8.617333e-5
	tt = kbz*np.array([300,600,900])
	m = -0.02
	for rd in radix:	
		e = har*energie[rd] 
		dd = dos[rd]/har/volume
		symb = ['--','-.','-']
		scala=1e-21
		for ind, t in enumerate(tt):
			nofE = dd/(np.exp((e-m)/t)+1.)
			sy=symb[ind]
			ax2.plot (nofE*scala,e,sy,label='T='+str(int(t/kbz))+' K')
			ax2.legend(fontsize=legsize)
	ax2.tick_params(right= False ,top= False,left= True, bottom= True)
	ax2.set_yticklabels([]) 
	ax2.set_xlabel('n(E) (10$^{21}$ cm$^{-3}$)',fontsize=labsize)
	ax2.set_ylim(m,0.3)
	ax2.set_xlim(0,1e21*scala)


	ax2.set_ylabel('n(E) (cm$^{-3}$)')
	ax2.yaxis.set_label_position("right")
#
	dum=input("\nMax points/segment (enter: default=100) => ")
	if dum=='':
		nkmax=100
	else:
		nkmax=int(dum)
	
	kdef="[[0,0,0],[0.5,0,0],[0.25,0.25,0],[0,0,0]]"
	kpath=input(f'\nPath (enter: default={kdef} ) => ')
	if kpath=='': kpath=kdef

	print(f'\nBands plot:\nMax pts/segment:{nkmax}\nPath:{kpath}')

	e_bk={}
	kpaths = ast.literal_eval(kpath)
	kstr = [str(x) for x in kpaths]
	kpaths = [list(gx)for k,gx in itertools.groupby(kpaths, key=lambda x: x is not None) if k]
	kpaths = [np.array(i, dtype=np.float64) for i in kpaths]
	
	shifta=2.60

	for ikpath, kpath in enumerate(kpaths):

		actpt=np.shape(kpath)[0]-1
		#print(f'ikpath={ikpath}\nkpath=\n{kpath}\nshape-kpath={np.shape(kpath)}\nactpt={actpt}')

		dl=np.zeros((actpt,3))

		lenp = np.zeros(actpt)
		nkpt = np.zeros(actpt)
	
		for ipt in range(actpt):
			dum=0
			for j in range(3):
				dum += (kpath[ipt+1,j]-kpath[ipt,j])**2
			lenp[ipt]=np.sqrt(dum)

		lenp=lenp/max(lenp)
		nkpt=np.array([int(nkmax*y) for y in lenp])
                
		print(f'N of segments: {actpt}\nN of pts/segment: {nkpt}')

		for ipt in range(actpt):
			for j in range(3):
				dl[ipt,j] = (kpath[ipt+1,j]-kpath[ipt,j])/(nkpt[ipt]-1)

		nkpoints = sum(nkpt)
		kp=np.zeros((nkpoints,3))
		offst=0
		for ipt, pt in enumerate(nkpt):
			for i in range(pt):
				for j in range(3):
					kp[offst+i,j]=kpath[ipt,j]+dl[ipt,j]*i
			offst += pt 

		pathb = asekp.bandpath(kpath, data.atoms.cell, nkpoints)
		dkp, dcl, dum = pathb.get_linear_kpoint_axis()
			
		egrid = getBands(kp, equivalences, data.get_lattvec(), coeffs)[0]
		egrid -= data.fermi
		nbands = egrid.shape[0]
		
		ax.set_prop_cycle(color=matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])
		
		ticdiv=[dkp[0]]
		ind = 0
		for ikp in nkpt:
			ind += ikp
			ticdiv.append(dkp[ind-1])
		ticdiv=np.asarray(ticdiv)

		for i in range(nbands):
			ax.plot(dkp, -shifta +har*egrid[i, :], lw=2.)
		
		ax.set_xticks(ticdiv)
		kstr=['',r'$\leftarrow k_z\ \ \   \Gamma\ \ \   k_y \rightarrow$','']
		ax.set_xticklabels(kstr)
		for d in ticdiv:
			ax.axvline(x=d, ls="--", lw=1)
		
		ax.set_ylabel(r"$\varepsilon - \mu\;\left[\mathrm{eV}\right]$",fontsize=labsize)	
#		ax.set_xlabel('k-path')

		plt.tight_layout(0.1)
	ax.set_ylim(0,0.3)

	kdef="[[0,0,0],[0.5,0,0],[0.25,0.25,0],[0,0,0]]"
	kpath=input(f'\nPath (enter: default={kdef} ) => ')
	if kpath=='': kpath=kdef

	print(f'\nBands plot:\nMax pts/segment:{nkmax}\nPath:{kpath}')

	e_bk={}
	kpaths = ast.literal_eval(kpath)
	kstr = [str(x) for x in kpaths]
	kpaths = [list(gx)for k,gx in itertools.groupby(kpaths, key=lambda x: x is not None) if k]
	kpaths = [np.array(i, dtype=np.float64) for i in kpaths]

	for ikpath, kpath in enumerate(kpaths):

		actpt=np.shape(kpath)[0]-1
		#print(f'ikpath={ikpath}\nkpath=\n{kpath}\nshape-kpath={np.shape(kpath)}\nactpt={actpt}')

		dl=np.zeros((actpt,3))

		lenp = np.zeros(actpt)
		nkpt = np.zeros(actpt)
	
		for ipt in range(actpt):
			dum=0
			for j in range(3):
				dum += (kpath[ipt+1,j]-kpath[ipt,j])**2
			lenp[ipt]=np.sqrt(dum)

		lenp=lenp/max(lenp)
		nkpt=np.array([int(nkmax*y) for y in lenp])
                
		print(f'N of segments: {actpt}\nN of pts/segment: {nkpt}')

		for ipt in range(actpt):
			for j in range(3):
				dl[ipt,j] = (kpath[ipt+1,j]-kpath[ipt,j])/(nkpt[ipt]-1)

		nkpoints = sum(nkpt)
		kp=np.zeros((nkpoints,3))
		offst=0
		for ipt, pt in enumerate(nkpt):
			for i in range(pt):
				for j in range(3):
					kp[offst+i,j]=kpath[ipt,j]+dl[ipt,j]*i
			offst += pt 

		pathb = asekp.bandpath(kpath, data.atoms.cell, nkpoints)
		dkp, dcl, dum = pathb.get_linear_kpoint_axis()
			
		egrid = getBands(kp, equivalences, data.get_lattvec(), coeffs)[0]
		egrid -= data.fermi
		nbands = egrid.shape[0]
		
		ax3.set_prop_cycle(color=matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])
		
		ticdiv=[dkp[0]]
		ind = 0
		for ikp in nkpt:
			ind += ikp
			ticdiv.append(dkp[ind-1])
		ticdiv=np.asarray(ticdiv)

		for i in range(nbands):
			ax3.plot(dkp, -shifta + har*egrid[i, :], lw=2.)
		
		ax3.set_xticks(ticdiv)
		kstr=['',r'$\leftarrow k_y\ \ \ \Gamma\ \ \  k_x \rightarrow$','']
		ax3.set_xticklabels(kstr)
		for d in ticdiv:
			ax3.axvline(x=d, ls="--", lw=1)
		
		#ax3.set_xlabel('k-path')

		plt.tight_layout(0.1)
	ax3.set_yticklabels([]) 
	ax3.set_ylim(0,0.3)


	plt.tight_layout(0.1)

	plt.show()
	endplot = sum(stuffstatus)<=1

if c_ebk:

	print('\nPlotting bands (method independent)\n')
	dum=input("\nMax points/segment (enter: default=100) => ")
	if dum=='':
		nkmax=100
	else:
		nkmax=int(dum)
	
	kdef="[[0,0,0],[0.5,0,0],[0.25,0.25,0],[0,0,0]]"
	kpath=input(f'\nPath (enter: default={kdef} ) => ')
	if kpath=='': kpath=kdef

	print(f'\nBands plot:\nMax pts/segment:{nkmax}\nPath:{kpath}')

	e_bk={}
	kpaths = ast.literal_eval(kpath)
	kstr = [str(x) for x in kpaths]
	kpaths = [list(gx)for k,gx in itertools.groupby(kpaths, key=lambda x: x is not None) if k]
	kpaths = [np.array(i, dtype=np.float64) for i in kpaths]

	plt.figure()
	ax = plt.gca()

	for ikpath, kpath in enumerate(kpaths):

		actpt=np.shape(kpath)[0]-1
		#print(f'ikpath={ikpath}\nkpath=\n{kpath}\nshape-kpath={np.shape(kpath)}\nactpt={actpt}')

		dl=np.zeros((actpt,3))

		lenp = np.zeros(actpt)
		nkpt = np.zeros(actpt)
	
		for ipt in range(actpt):
			dum=0
			for j in range(3):
				dum += (kpath[ipt+1,j]-kpath[ipt,j])**2
			lenp[ipt]=np.sqrt(dum)

		lenp=lenp/max(lenp)
		nkpt=np.array([int(nkmax*y) for y in lenp])
                
		print(f'N of segments: {actpt}\nN of pts/segment: {nkpt}')

		for ipt in range(actpt):
			for j in range(3):
				dl[ipt,j] = (kpath[ipt+1,j]-kpath[ipt,j])/(nkpt[ipt]-1)

		nkpoints = sum(nkpt)
		kp=np.zeros((nkpoints,3))
		offst=0
		for ipt, pt in enumerate(nkpt):
			for i in range(pt):
				for j in range(3):
					kp[offst+i,j]=kpath[ipt,j]+dl[ipt,j]*i
			offst += pt 

		pathb = asekp.bandpath(kpath, data.atoms.cell, nkpoints)
		dkp, dcl, dum = pathb.get_linear_kpoint_axis()
			
		egrid = getBands(kp, equivalences, data.get_lattvec(), coeffs)[0]
		egrid -= data.fermi
		nbands = egrid.shape[0]
		
		ax.set_prop_cycle(color=matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])
		
		ticdiv=[dkp[0]]
		ind = 0
		for ikp in nkpt:
			ind += ikp
			ticdiv.append(dkp[ind-1])
		ticdiv=np.asarray(ticdiv)

		for i in range(nbands):
			plt.plot(dkp, har*egrid[i, :], lw=2.)
		
		ax.set_xticks(ticdiv)
		ax.set_xticklabels(kstr)
		for d in ticdiv:
			plt.axvline(x=d, ls="--", lw=1)

		plt.ylabel(r"$\varepsilon - \varepsilon_F\;\left[\mathrm{eV}\right]$",fontsize=labsize)
		plt.tight_layout()
	plt.show()
	dum=input('Enter to continue: ')
        
plt.show()


def E_min_CB(ene,fermi0,dos):
        """
        gives the minimum energy of conduction band. if offset and skimming is applied before it gives the next smallest value of energy from CB.
        Input:  - list of energies 
                        - fermi energy
                        - list of density of states
        output flaot minimum from CB
        """
        auxiliary = ene[ np.logical_and(ene > fermi0 , dos!=0) ]
        
        return np.amin(auxiliary)
