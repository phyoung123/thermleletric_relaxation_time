#!/usr/bin/env python3


import numpy as np 
import sys
import os.path
import relaxTau as rt
from BoltzTraP2 import units
import codecs,json


# classe numpy encoder. server per salvare le array di numpy in file json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


### MAIN

# salva tau dal modello nel formato tau[T,E]

# parameters of calculation:
inputs = {'doping_tau_plt':'nan','n_ene_points':'nan','nTemperatures':'nan','minTemperature':'nan','maxTemperature':'nan'}

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

nEns = int(inputs['n_ene_points'])	# number of bins to suddivide energy ene in bins. 
nTemps = int(inputs['nTemperatures'])	# number of temperatures to check
minT = float(inputs['minTemperature'])	# lowest temperature
maxT = float(inputs['maxTemperature'])	# highest temperature


# energies in eV
minE = 1e-6
maxE = 0.5

doping = float(inputs['doping_tau_plt']) 	# unita: elettroni / m^-3

Tmps = np.linspace(minT, maxT, num=nTemps)
mu = 0.
Ens = np.linspace(minE, maxE, num=nEns)



tau_ac = [[rt.tau_ac(t,e*units.eV/units.Joule) for e in Ens] for t in Tmps]
tau_ac = np.asarray(tau_ac)

tau_imp = [[rt.tau_imp(t,e*units.eV/units.Joule,doping) for e in Ens] for t in Tmps]
tau_imp = np.asarray(tau_imp)

tau_pol = [[rt.tau_pol(t, mu, e*units.eV/units.Joule, 1) for e in Ens] for t in Tmps]
tau_pol = np.asarray(tau_pol)

tau_tot = np.zeros_like(tau_ac)
for iT in range(len(tau_ac)):
	for iE in range(len(tau_ac[0])):
		tau_tot[iT,iE] = 1/(1/tau_ac[iT,iE] + 1/tau_imp[iT,iE] + 1/tau_pol[iT,iE])




datadir_out = os.path.abspath(os.path.dirname(__file__))
datadir_out = os.path.abspath(os.path.join(datadir_out, '../data/'))


calcData = {}
calcData['tau_ac'] = tau_ac
calcData['tau_imp'] = tau_imp
calcData['tau_pol'] = tau_pol
calcData['tau_tot'] = tau_tot
calcData['Ens'] = Ens
calcData['Tmps'] = Tmps

for datakey in calcData.keys():
	outputNameJSON = datadir_out + '/{}.json'.format(datakey)
	with open(outputNameJSON,'w') as fil:
		json.dump(calcData[datakey], fil, cls=NumpyEncoder)
	print('saved ', datakey,' to file')














