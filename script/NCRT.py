#!/usr/bin/env python3

import math as mt
import numpy as np
import os.path
import copy
import sys
from scipy import integrate

import codecs,json

from BoltzTraP2.misc import info
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


# classe numpy encoder. server per salvare le array di numpy in file json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# funzione per ricavare l'indice del vettore k
def indK(k,dim):
	"""Gives the index of a k-point.calcData
	Input: (k-point, vector dims)
	output: integer
	vector dims is a vector with the number of divisions along each axis in reciprocal lattice"""
	return (mt.floor(dim[0]/2.) + k[0])*dim[1]*dim[2] + (mt.floor(dim[1]/2.) + k[1])*dim[2] + mt.floor(dim[2]/2.) +k[2]


# get index of e_bk for wich e_bk > fermi0
def index_grtr_ef(ebk,fermi0):
	"""
	gives index of energies of e_bk greater than fermi energy.
	input: e_bk (interpolated) , fermi energy
	output list of indices [i,j,k]
	"""
	return np.argwhere(ebk>fermi0)
	


# get minimum energy of conduction band E_min_CB
def E_min_CB(ene,fermi0,dos):
	"""
	gives the minimum energy of conduction band. if offset and skimming is applied before it gives the next smallest value of energy from CB.
	Input: 	- list of energies 
			- fermi energy
			- list of density of states
	output flaot minimum from CB
	"""
	auxiliary = ene[ np.logical_and(ene > fermi0 , dos!=0) ]
	
	return np.amin(auxiliary)





# main ----------------------------------------------------------------

# parameters of calculation:

inputs = {'multiplier':'nan','n_ene_points':'nan','nTemperatures':'nan','n_mu_points':'nan','minTemperature':'nan','maxTemperature':'nan','ncores':'nan','radixType':'nan','latticeThermalConductivity':'nan','phonon_replicae':'nan','energy_range':'nan','scissor':'nan'}



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

mult = float(inputs['multiplier']) # multiplicative factor for k-points grid
npts = int(inputs['n_ene_points'])	# number of bins to suddivide energy ene in bins. 
nTemps = int(inputs['nTemperatures'])	# number of energy to check
n_mu_pts = int(inputs['n_mu_points']) # number of divisions for chemical potential mu
minT = float(inputs['minTemperature'])	# lowest temperature
maxT = float(inputs['maxTemperature'])	# highest temperature
ncores = int(inputs['ncores'])			# number of cores for multiprocessing
rep = int(inputs['phonon_replicae'])	# number of replicae of phonons
Har= 27.21386
scissor = float(inputs['scissor'])/Har      #scissor operator
Tr = np.linspace(minT, maxT, num=nTemps)

radixType = inputs['radixType']


kl = np.zeros((nTemps,3,3))

#convert input string to numpy array
auxiliary_kl = np.array( json.loads(inputs['latticeThermalConductivity']) )

Tdep = np.trace(auxiliary_kl)
if Tdep<0:
    print('\nLattice k: 1/T dependence turned on\n')
    for iT,vT in enumerate(Tr):
        kl[iT] = -300/vT * auxiliary_kl 
elif Tdep==0:
    print('\nLattice k: read from file\n')
    with open('../input/latthcond','r') as f:
        for iT,vT in enumerate(Tr):
            kl[iT] = float(f.readline())
else:
    print('\nConstant lattice k\n')
    for iT,vT in enumerate(Tr):
        kl[iT] = auxiliary_kl  


# zero off-diag kl set to 1e-10 (to avoid divide-by-zero in ZT below)
for i_T,val_T in enumerate(Tr):
	for i,vi in enumerate(kl[i_T]):
		for j,vj in enumerate(vi):
			if vj == 0 and i != j:
				kl[i_T,i,j] = 1e-10

# read data from vasp dft file. 
datadir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.abspath(os.path.join(datadir, "../"+'input' ))

radName = radixType+'_btp'
try:
	data = BTP.DFTData(datadir)
	print("loading dft data.")
except:
	print("error! unable to load dft data from folder",datadir)

range_in = np.array(json.loads(inputs['energy_range']))
print(f'Before chop-off:\nNelect: {int(data.nelect)}')
print(f'Empty bands:{int(2*len(data.bandana())-data.nelect)}')
if any(range_in):
	data.bandana(range_in[0],range_in[1])[0]
	print(f'After chop-off:\nNelect: {int(data.nelect)}')
	print(f'Empty:{int(2*len(data.bandana())-data.nelect)}')	
else:
	print(f'No band chop-off.')

# generate irriducibile k-points mesh from primitive cell in
# data.atoms, kpoints are normalized: the biggest integer is mult*len(data.kpoints)
equivalences = sphere.get_equivalences( data.atoms, data.magmom, mult * len(data.kpoints) ) 
print( "Generated",len(equivalences),"irreducible k-points.")

#apply scissor operator - NB only valid for insulators

data.ebands[ data.ebands > data.fermi ] += scissor

## compute above mentioned coefficients
try:
	coef = fite.fitde3D(data,equivalences)
	print('computed coefficients (1)')
	serialization.save_calculation(radName + str(mult) + '.bt2',data, equivalences, coef, serialization.gen_bt2_metadata(data, data.mommat is not None) )
	print('saved interpolation to ' + radName + str(mult) + '.bt2')
except:
	print("error! unable compute or store coefficients (1)")

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

# compute e_b(k), vxv_b(k), curvature_b(k) ----

latVec = data.atoms.get_cell().T
# rende i vettori della cella primitiva
# altrimenti usa latVec = data.get_lattvec()

try:
	e_bk, vxv_bk, curv_bk = fite.getBTPbands(equivalences, coef, latVec, curvature=False, nworkers=ncores)
	print('calculated bands e_bk for every band and k-point')
except:
	print('error, unable to calculate bands e_bk')
#
# calcola i parametri di Onsager
## divide up energy in npts bins, calculate DOS for each bin.

ene, dos, vxvDos, curvDos = BL.BTPDOS( e_bk, vxv_bk, npts=npts )
print('computed DOS for every energy')


# calcola l'energia di fermi e la paragona a quella fornita da VASP
try:
	fermi0 = BL.solve_for_mu(ene, dos, data.nelect, 0, data.dosweight)
	fermi0VASP = data.fermi
	print('calculated fermi energy')
	print('chemical potential for T=0 is {} Ha'.format(fermi0))
	print('VASP fermi energy:', fermi0VASP)
except:
	print('error while calculating fermi energy')


# translate all energies so that bottom of conduction band is zero

offset = E_min_CB(ene,fermi0,dos) 
print('ECB,fermi',offset,fermi0)
ene = ene - offset
fermi0 = fermi0 - offset
fermi0VASP = fermi0VASP - offset
e_bk = e_bk - offset
data.ebands = data.ebands - offset

print('applied offset. New fermi energy = {} Ha'.format(fermi0))

#del e_bk # this is for saving ram. comment if you want e_bk saved to file

## generate list of temperatures and chemical potentials 

mu_min = E_min_CB(ene,fermi0,dos) - 3.5 * units.BOLTZMANN * maxT
mu_max = E_min_CB(ene,fermi0,dos) + 3 * units.BOLTZMANN * maxT
mu = np.linspace(mu_min, mu_max, num=n_mu_pts)

# lista di Doping[temperatura][potenziale chimico]
# units: n electrons/cm^3

N0 = data.nelect
Doping = np.array([[(N0 + BL.calc_N(ene,dos,potChim,tmp))/data.atoms.get_volume()*1e24  for potChim in mu] for tmp in Tr])
# in realta si può prenere anche da BL.fermiintegrals()[0]
print('calculated Doping for each T and mu')

# ***
## compute tau(T,mu,ene)
# -------------------------------------------------------------

tau = np.ones((len(Tr),len(mu),len(ene)))

for i_T,val_T in enumerate(Tr):
	for i_mu,val_mu in enumerate(mu):
		for i_en,v_en in enumerate(ene):
			if v_en > 0 and Doping[i_T][i_mu]<0:
				tau[i_T][i_mu][i_en] = relaxTau.tau_tot(val_T,val_mu/units.Joule,v_en/units.Joule,rep,abs(Doping[i_T][i_mu])*1e+6)
			else:
				tau[i_T][i_mu][i_en] = 0
				
print('computed tau for relevant temperature, chem potential and energies')

vxvDos_T_mu = np.zeros((3,3,len(Tr),len(mu),len(ene)))

#multiply vxv x tau
for i_T,val_T in enumerate(Tr):
	for i_mu,val_mu in enumerate(mu):
		for i_en,v_en in enumerate(ene):
			vxvDos_T_mu[:,:,i_T,i_mu,i_en] = vxvDos[:,:,i_en] * tau[i_T,i_mu,i_en]

## compute Onsager coefficients in the form sigma[temperature][chemical potential] --------------------------
N = np.zeros((len(Tr),len(mu)))
L0 = np.zeros((len(Tr),len(mu),3,3))
L1 = np.zeros((len(Tr),len(mu),3,3))
L2 = np.zeros((len(Tr),len(mu),3,3))

for i_T,val_T in enumerate(Tr):
	for i_mu,val_mu in enumerate(mu):
		N[i_T,i_mu], L0[i_T,i_mu], L1[i_T,i_mu], L2[i_T,i_mu], Lm11 = BL.fermiintegrals( ene, dos, vxvDos_T_mu[:,:,i_T,i_mu], mur=[val_mu], Tr=[val_T] )

sigma, seebeck, ke, hall = BL.calc_Onsager_coefficients(L0, L1, L2, mu, Tr, data.atoms.get_volume())
print('calculated Onsager coefficients for every T and mu.')



# compute traces for sigma, seebeck, ke -----------------------------------------------------------------------------------
sigma_tr = np.array([[ sigma[tmp][pchm][0][0] + sigma[tmp][pchm][1][1] + sigma[tmp][pchm][2][2] for pchm in range(len(mu)) ]for tmp in range(len(Tr))])/3

seebeck_tr = np.array([[ seebeck[tmp][pchm][0][0] + seebeck[tmp][pchm][1][1] + seebeck[tmp][pchm][2][2] for pchm in range(len(mu)) ]for tmp in range(len(Tr))])/3

ke_tr = np.array([[ ke[tmp][pchm][0][0] + ke[tmp][pchm][1][1] + ke[tmp][pchm][2][2] for pchm in range(len(mu)) ]for tmp in range(len(Tr))])/3

print('computed traces of sigma, seebeck, ke')


# compute ZT and optimal mu for wich ZT is max
ZT = np.empty_like(sigma)
for i_T,val_T in enumerate(Tr):
	for i_mu,v_mu in enumerate(mu):
		ZT[i_T,i_mu] = sigma[i_T,i_mu]*seebeck[i_T,i_mu]**2/(ke[i_T,i_mu] + kl[i_T]) * val_T
print('computed ZT')

ZT_tr = np.array([[ ZT[tmp][pchm][0][0] + ZT[tmp][pchm][1][1] + ZT[tmp][pchm][2][2] for pchm in range(len(mu)) ]for tmp in range(len(Tr))])/3

# find index of optimal chemical potential for which ZT_tr is maximum for a given T.
optimal_mu = np.zeros(len(Tr))
optimal_mu = np.argmax(ZT_tr, axis=1)
print('computed trace of ZT')




# salva i dati in un file json. con la classe NumpyEncoder jsonizza le numpy.ndarray
# i formati sono i seguenti: 
# k[i] per un totale di len(dims) k-points. 
# e_bk[b][i] = energy of band b at k-point i-th
# dims[j]= numero di punti nella k-griglia lungo la direzione j. 
# ene[l] = energia associata al bin numero l
# dos[l] = dos associato al bin l di energia ene[l].
# Temp[i] = array di temperature mu[j] array di potenziali chimici
# N[T][mu] = electron count at T and mu, Doping[T][mu] = N0 + N[T][mu]. N0 è il conteggio di elettroni di vasp, quando il pot chimico è l'energia di fermi e la temperatura è zero. cioè N0 = N[T=0][mu=Ef]
# sigma[T][mu], seebeck[T][mu], ke[T][mu], doping[T][mu] quantities after integration, depending on T and mu


calcData = {}
calcData['fermi_energy'] = np.array(fermi0)
calcData['fermi_vasp'] = np.array(fermi0VASP)
calcData['vasp_ebands'] = data.ebands
calcData['kpoints'] = kpnts
calcData['e_bk'] = e_bk
calcData['energie'] = ene
calcData['dallvec'] = dallvec
calcData['dos'] = dos
calcData['Temp'] = Tr
calcData['chemPot'] = mu
calcData['Nelect'] = N
calcData['sigma'] = sigma
calcData['seebeck'] = seebeck
calcData['ke'] = ke
calcData['sigma_tr'] = sigma_tr
calcData['seebeck_tr'] = seebeck_tr
calcData['ke_tr'] = ke_tr
calcData['doping'] = Doping
calcData['dims'] = dims
calcData['ZT'] = ZT
calcData['ZT_tr'] = ZT_tr
calcData['optimal_mu'] = optimal_mu

	
calcData_name_shape = [[ky, calcData[ky].shape] for ky in calcData.keys()]

calcData['info'] = 'CRT. this file contains the following data (shapes):\nthere are: {} total k-points, {} energies, {} temperatures form {} to {} K, and {} chemical potentials.\nfermi energy is = {} Ha\nVolume is {}\n'.format( len(calcData['kpoints']), len(calcData['energie']), len(calcData['Temp']), calcData['Temp'].min(), calcData['Temp'].max(), len(calcData['chemPot']), calcData['fermi_energy'], data.atoms.get_volume())


for dts in calcData_name_shape:
	calcData['info'] = calcData['info'] + '\n' + str(dts[0]) + '\t' + str(dts[1])

datadir_out = os.path.abspath(os.path.dirname(__file__))
datadir_out = os.path.abspath(os.path.join(datadir_out, '../data/'))

for datakey in calcData.keys():
	outputNameJSON = datadir_out + '/' + radixType + '_btp-NCRT-{}-{}-{}-{}-{}.json'.format(datakey,mult,npts,nTemps,n_mu_pts)
	with open(outputNameJSON,'w') as fil:
		json.dump(calcData[datakey], fil, cls=NumpyEncoder)
	print('saved ', datakey,' to file')


print('\nsaved data to files '+radixType+'_btp-NCRT-{}-{}-{}-{}-{}.json'.format(datakey,mult,npts,nTemps,n_mu_pts))
print(calcData['info'])












































