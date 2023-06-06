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

verysmall=1e-10

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
Har = 27.21386
scissor = float(inputs['scissor'])/Har     #scissor operator
print(f'scissor (Har) = {scissor}')


Tr = np.linspace(minT, maxT, num=nTemps)

radixType = inputs['radixType']

kl = np.zeros((nTemps,3,3))

#convert input string to numpy array
auxiliary_kl = np.array( json.loads(inputs['latticeThermalConductivity']) )
auxiliary_kl += verysmall

Tdep = np.trace(auxiliary_kl)

if Tdep<0:
    print('\nLattice k: 1/T dependence turned on\n')
    for iT,vT in enumerate(Tr):
        kl[iT] = -300/vT * auxiliary_kl 
elif np.isclose(Tdep,0.0):
    print('\nLattice k: read from file\n')
    with open('../input/latthcond','r') as f:
        for iT,vT in enumerate(Tr):
            kl[iT] = float(f.readline())
else:
    print('\nConstant lattice k\n')
    for iT,vT in enumerate(Tr):
        kl[iT] = auxiliary_kl  


# set to 1e-10 values of kl which are zero outside the diagonal (this is because you cannot divide by zero when computin ZT below)
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
if any(range_in):
	data.bandana(range_in[0],range_in[1])[0]
	print(f'After chop-off:\nNelect: {int(data.nelect)}')
else:
	print(f'No band chop-off.')

## generate irriducibile k-points mesh from primitive cell in data.atoms, kpoints are normalized: the biggest integer is mult*len(data.kpoints)
equivalences = sphere.get_equivalences( data.atoms, data.magmom, mult * len(data.kpoints) ) 
print( "Generated",len(equivalences),"irreducible k-points.")

#add scissor
data.ebands[ data.ebands > data.fermi ] += scissor

## compute above mentioned coefficients
try:
	coef = fite.fitde3D(data,equivalences)
	print('computed coefficients (1)')
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



# compute e_b(k), vxv_b(k), curvature_b(k) -----------------------------------------------------------------------
latVec = data.atoms.get_cell().T # rende i vettori della cella primitiva  altrimenti usa latVec = data.get_lattvec()
try:
	e_bk, vxv_bk, curv_bk = fite.getBTPbands(equivalences, coef, latVec, curvature=False, nworkers=ncores)
	print('calculated bands e_bk for every band and k-point')
except:
	print('error, unable to calculate bands e_bk')

# calcola i parametri di Onsager
## begin with dividing energies in npts different bins, and calculate density of states for each bin.

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
#del e_bk ## this is for saving ram. comment if you want e_bk saved to file

## generate a list of temperatures and chemical potentials ----------------------------------------------------------------------------
## generate a list of chemical potential.
mu_min = E_min_CB(ene,fermi0,dos) - 3.5 * units.BOLTZMANN * maxT
mu_max = E_min_CB(ene,fermi0,dos) + 3 * units.BOLTZMANN * maxT
mu = np.linspace(mu_min, mu_max, num=n_mu_pts)


#i_EnFer = (np.fabs(mu-fermi0)).argmin(axis=0) # index of mu closest to fermi0

# crea una lista di Doping: Doping[temperatura][potenziale chimico] units: n electrons/cm^3
N0 = data.nelect
Doping = np.array([[(N0 + BL.calc_N(ene,dos,potChim,tmp))/data.atoms.get_volume()*1e24  for potChim in mu] for tmp in Tr]) # in realta si può prenere anche da BL.fermiintegrals()[0]
print('calculated Doping for each T and mu')


## compute Onsager coefficients in the form sigma[temperature][chemical potential] --------------------------
try:
	N, L0, L1, L2, Lm11 = BL.fermiintegrals( ene, dos, vxvDos, mur=mu, Tr=Tr )
	sigma, seebeck, ke, hall = BL.calc_Onsager_coefficients(L0, L1, L2, mu, Tr, data.atoms.get_volume())
	print('calculated Onsager coefficients for every T and mu.')
except:
	print('error, couldn\'t compute Onsager coefficients')

# getting the relevant relaxation time averages over energies
#tau_ave = relaxTau.tau_ave(Tr,mu/units.Joule,ene/units.Joule,dos/units.Joule)

#tau_cum = np.array( [ [ relaxTau.tau_cum_T(tmp,potchem/units.Joule,ene/units.Joule,dos*units.Joule) for potchem in mu ] for tmp in Tr ] )


ene_pos = relaxTau.positive_energies(ene)
tau_ave = np.zeros((len(Tr),len(mu)))


# compute tau_ave and tau_cum
temps_splitted = np.array_split(Tr,ncores)

ques = [mp.Queue() for _ in temps_splitted]
jobs = [mp.Process(target=relaxTau.tau_cum_T_mp, args=(tmps, mu/units.Joule, ene/units.Joule, dos*units.Joule,abs(Doping)*1e+6,rep,ques[i_tmps])) for i_tmps,tmps in enumerate(temps_splitted)]

for jb in jobs: jb.start()

tau_cum = ques[0].get()
for q in range(1,len(ques)):
	tau_cum = np.append(tau_cum,ques[q].get(),axis=0)
	
for jb in jobs: jb.join()


for i_T,val_T in enumerate(Tr):
	for i_mu,val_mu in enumerate(mu):
		tau_ave[i_T][i_mu] = tau_cum[i_T,i_mu, len(tau_cum[i_T,i_mu]) - 1 ]

# multiplyng with above value to get actual sigma and ke
for i_T,val_T in enumerate(Tr):
	for i_mu,val_mu in enumerate(mu):
		sigma[i_T,i_mu] = sigma[i_T,i_mu] * tau_ave[i_T,i_mu]
		ke[i_T,i_mu] = ke[i_T,i_mu] * tau_ave[i_T,i_mu]

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
calcData['tau_ave'] = tau_ave
calcData['tau_cum'] = tau_cum
calcData['ene_pos'] = ene_pos
	
calcData_name_shape = [[ky, calcData[ky].shape] for ky in calcData.keys()]

calcData['info'] = 'CRT. this file contains the following data (shapes):\nthere are: {} total k-points, {} energies, {} temperatures form {} to {} K, and {} chemical potentials.\nfermi energy is = {} Ha\nVolume is {}\n'.format( len(calcData['kpoints']), len(calcData['energie']), len(calcData['Temp']), calcData['Temp'].min(), calcData['Temp'].max(), len(calcData['chemPot']), calcData['fermi_energy'], data.atoms.get_volume())


for dts in calcData_name_shape:
	calcData['info'] = calcData['info'] + '\n' + str(dts[0]) + '\t' + str(dts[1])

datadir_out = os.path.abspath(os.path.dirname(__file__))
datadir_out = os.path.abspath(os.path.join(datadir_out, '../data/'))

for datakey in calcData.keys():
	outputNameJSON = datadir_out + '/' + radixType + '_btp-ART-{}-{}-{}-{}-{}.json'.format(datakey,mult,npts,nTemps,n_mu_pts)
	with open(outputNameJSON,'w') as fil:
		json.dump(calcData[datakey], fil, cls=NumpyEncoder)
	print('saved ', datakey,' to file')


print('\nsaved data to files '+radixType+'_btp-ART-{}-{}-{}-{}-{}.json'.format(datakey,mult,npts,nTemps,n_mu_pts))
print(calcData['info'])

from BoltzTraP2 import serialization
serialization.save_calculation('bt2_file',data, equivalences, coef, serialization.gen_bt2_metadata(data, data.mommat is not None) )





































