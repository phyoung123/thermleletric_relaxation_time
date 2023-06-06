from BoltzTraP2 import units 
import math, sys, warnings
import numpy as np
from scipy import integrate
import BoltzTraP2.fd as fd
import sys
import json


# library for calculating relaxation time tau(T,mu,E). It uses SI units allover.


el=1.602e-19; ev2j=el;tpi=2*np.pi;fpi=4*np.pi
hbar=1.05457148e-34; kb=1.38064852e-23; 
eps0=8.85e-12; me=9.10938188e-31;mev2Hz=2.42e11   

# questa riga è opzionale, è comoda per ricognizione kdevelop
#global ni, epslatx, epslaty, epslatz, epsinf, epslat, eps, Zi, mav, D, rho, vsound, LO, uepsp

#seguono i parametri caratteristici per il SNO
#def parameters():
	## parameters for SNO
	#global ni, epslatx, epslaty, epslatz, epsinf, epslat, eps, Zi, mav, D, rho, vsound, LO, uepsp
	#ni = 5e26						# concentrazione di impurità ionizzate cm^-3
	#epslatx=33.5;epslaty=39.7;epslatz=133.9		# dielectric tensor of SNO, data gathered from materialsproject.org (tensor is diagonal)
	#epsinf=4.7*eps0					# eps(inf): valore a cui tende eps(omega) per omega = infinity
									## di queste tre epsilon useremo solo la più piccola, cioe epslatx *** (è quella che fornisce un contributo maggiore a P_imp)
	#epslat=epslatx*eps0				# costante dielettrica (stiamo moltiplicando solo per epslatx)
	#eps = epslat + epsinf			
	#Zi = 1.							# impurity charge, assuming 1
	#mav = 0.3 * me					# average electron mass
	#D=10*ev2j						# deformation potential of band energies, calculated at band extrema
	#rho = 4.93e3					# kg/m^3 density of material, data from materialsproject.org
	#vsound = 4438					# m/s data from 'anisotropic thermal...' DOI: 10.1111/j.1551-2916.2009.03533.x
	#TO = np.array([0.004])				# eV. transverse optical phonon. quantum of energy in scattering processes. It is an array, it is generalizable to the case where multiple scattering occurs.
	#LO = ev2j*TO*np.sqrt(eps/epsinf)	# energy for Longitudinal Optical phonon, still an array, as above (for TO).
	#uepsp = 1/epsinf-1/eps 			# convenient expression used in formula for W0 in Z function.
	
	
# -------------------------------------------------------------------

inpu = {'permit_infty':'nan','eps_lattice':'nan','impurity_charge':'nan','mass_e_avg':'nan','deformation_potential':'nan','density':'nan','speed_sound':'nan','LO_energies':'nan'}


with open('../input/input','r') as in_file:
	for line in in_file:
		words = line.split()
		for param in inpu:
			if len(words) != 0:
				if words[0] == param:
					for i_wrd,wrd in enumerate(words):
						if wrd == '=':
							inpu[param] = words[i_wrd+1]

# checking for errors in reading inputs
for i in inpu:
	if inpu[i] == 'nan':
		print('error! unable to load ',i,'. check input file for syntax. exiting')
		sys.exit()

epsinf=eps0 * float(inpu['permit_infty'])		# eps(inf): valore a cui tende eps(omega) per omega = infinity
epslat=eps0 * float(inpu['eps_lattice'])		# costante dielettrica (stiamo moltiplicando solo per epslatx)
eps = epslat + epsinf			
Zi = float(inpu['impurity_charge'])				# impurity charge, assuming 1
mav = me * float(inpu['mass_e_avg'])					# average electron mass
D=ev2j * float(inpu['deformation_potential'])	# deformation potential of band energies, calculated at band extrema
rho= float(inpu['density'])							# kg/m^3 density of material, data from materialsproject.org
vsound= float(inpu['speed_sound'])						# m/s data 
uepsp = 1/epsinf-1/eps 			# convenient expression used in formula for W0 in Z function.



LO = ev2j * np.array( json.loads(inpu['LO_energies']) ) # energy for Longit. Opt phonon, an array.




print('INPUTS (tau)')
for i in inpu:
	print(i,inpu[i])
print()



def fermi(e,t,mu):
	"""fermi function, inputs: (Energy, Temperature, chemical potential). Returns float """

	f = 1./(np.exp((e-mu)/(kb*t))+1.)
	return f

def planck(ep,t):
	"""Bose-Einstein function, inputs: (phonon Energy, Temperature). Returns float """
	return 1./(np.exp(ep/(kb*t))-1.)

def A(t,mu,e,ep):
	"""One of four functions: A,B,C,Z, Input: (temperature, chemical potential, electron Energy, phonon energy). output: float.
	Only positive energies (sqrt problem) !"""
	pp=planck(ep,t); ff=fermi(e+ep,t,mu); f=fermi(e,t,mu)
	if (f==0. and ff==0.): # sometimes fermi(e,t,mu) overflows and f=0 and ff=0 in that case the division f/ff is equal to 1.
		a = (pp+1)*((2*e+ep)*np.arcsinh(np.sqrt(e/ep))-np.sqrt(e*(e+ep)))
	else:
		a = (pp+1)*ff/f*((2*e+ep)*np.arcsinh(np.sqrt(e/ep))-np.sqrt(e*(e+ep)))
	return a

def B(t,mu,e,ep):
	"""One of four functions: A,B,C,Z. Input: (temperature, chemical potential, electron Energy, phonon energy). output: float.
	Only positive energies (sqrt problem) !"""
	dum=0
	if e>ep:
		pp=planck(ep,t); ff=fermi(e-ep,t,mu); f=fermi(e,t,mu)
		if (f==0. and ff==0.):
			dum=pp*  1 *((2*e-ep)*np.arccosh(np.sqrt(e/ep))-np.sqrt(e*(e-ep)))
		else:
			dum=pp*ff/f*((2*e-ep)*np.arccosh(np.sqrt(e/ep))-np.sqrt(e*(e-ep)))
	return dum

def C(t,mu,e,ep):
	"""One of four functions: A,B,C,Z. Input: (temperature, chemical potential, electron Energy, phonon energy). output: float.
	Only positive energies (sqrt problem) !"""
	pp=planck(ep,t); ff=fermi(e+ep,t,mu); f=fermi(e,t,mu)
	if (f==0. and ff==0.):
		dum=(pp+1.)* 1  *np.arcsinh(np.sqrt(e/ep))
	else:
		dum=(pp+1.)*ff/f*np.arcsinh(np.sqrt(e/ep))
	if e > ep:
		ff=fermi(e-ep,t,mu)
		if (f==0. and ff==0.):
			dum+=pp* 1  *np.arccosh(np.sqrt(e/ep))
		else:
			dum+=pp*ff/f*np.arccosh(np.sqrt(e/ep))
	return 2*e*dum

def Z(t,ep):

	"""One of four functions: A,B,C,Z. Input: (temperature, phonon energy). output: float."""
	W0=el**2/(fpi*hbar)*np.sqrt(2*mav*ep/hbar**2)*uepsp
	return 2/(W0*np.sqrt(ep))
		


def P_pol(t,mu,e,rr):
	"""Probability/time for polar electron-polar optical phonon scattering.
	Input: (Temperature, chemical potential, electron energy, number of replicae)
	Only positive energies (sqrt problem) !
	use SI units!
	"""
	zilch=1e-6
	Pacc=zilch
	if uepsp<zilch: return Pacc
	repliche=[l+1 for l in range(rr)]
	for i in range(len(LO)): #ciclo su fononi e repliche
		for r in repliche:
			ep=r*LO[i]
			dummy=Z(t,ep)            
			if dummy<zilch:
				P=zilch
			else:
				P = (C(t,mu,e,ep)- A(t,mu,e,ep) - B(t,mu,e,ep)) / (dummy*e**1.5)
			Pacc += P
	return Pacc

def P_imp(t,e,dop):		
	"""Probability/time e-impurities scattering.
	Input (Temperature, Energy). Returns float. 
	Only positive energies (sqrt problem) !
	use SI units!"""
	q0=np.sqrt(el**2*dop/(eps*kb*t))
	dum=np.pi*Zi**2*dop*el**4/(e**1.5*np.sqrt(2*mav)*(fpi*eps)**2)
	x=(hbar*q0)**2/(8*mav*e)
	dum*=(np.log(1+1./x)-1./(1+x)) 
	return max(dum, 1e-6)

def P_ac(t,e):
	"""Probability/time acoustic phonon-electron scattering.
	Input (Temperature, electron Energy). Returns float. 
	Only positive energies (sqrt problem) !
	use SI units!
	"""
	dum=(2*mav)**1.5*np.sqrt(e)*(kb*t*D*D)/(tpi*hbar**4*rho*vsound**2)
	return max(dum,1e-6)

def tau_ac(t,e):
	"""relaxation time for electron-acoustic phonon scattering
	Input: (temperature, electron energy)
	use SI units!"""
	#parameters()
	return 1./P_ac(t,e)

def tau_imp(t,e,dop):
	"""relaxation time for electron-impurity scattering
	Input: (temperature, electron energy,doping)
	use SI units!"""
	#parameters()
	return 1./P_imp(t,e,dop)
	
def tau_pol(t,mu,e,rr):
	"""relaxation time for electron-polar optical phonon scattering
	Input: (temperature, chemical potential,electron energy, number of replicae)
	use SI units!"""
	#parameters()
	return 1./P_pol(t,mu,e,rr)

def tau_tot(t,mu,e,rr,dop):
	"""total relaxation time for electron scattering. It is the harmonic mean of tau_ac, tau_imp, tau_pol
	Input: (temperature, chemical potential, electron energy, number of replicae,doping)
	use SI units! make sure doping is positive!"""
	return 1./(1./tau_imp(t,e,dop)+1./tau_ac(t,e)+1./tau_pol(t,mu,e,rr))



# compute tau(T) at fermi level. value is energy-averaged. only positive energies are counted. 
def tau_cum_T(temp,chempot,ene,dos,dop,rep):
	"""
	compute mean relaxation time across all energies. it is given by (5.27) of fund of semiconductors yu,cardona 4th ed. use SI UNITS!!
	input:
	- 1 temperature
	- 1 fermi energy
	- array of energies
	- array of dos
	- number of replicae for phonon
	output:
	- array of cumulative tau_ave_T  i.e. integral of tau up to energy e_pos
	"""
	
	# get only the positive energies, and associated dos, and fermi function values, and tau_T(E) values
	
	energies_pos_indx = [ene > 0]
	energies_pos = ene[tuple(energies_pos_indx)]
	den_o_s_pos = dos[tuple(energies_pos_indx)]
	fermi_f = np.zeros_like(energies_pos)
	tau_T = np.zeros_like(energies_pos)
	dfde = np.zeros_like(energies_pos)
	
	
	
	for i_en,val_en in enumerate(energies_pos):
		fermi_f[i_en] = fermi(val_en,temp,chempot)
		tau_T[i_en] = tau_tot(temp,chempot,val_en,rep,dop)
	
	dfde = fd.dFDde(energies_pos,chempot,units.BOLTZMANN_SI*temp)
	
	f_integrand_num = 2./3. * tau_T * energies_pos * den_o_s_pos * (-dfde)
	#f_integrand_num = tau_T * den_o_s_pos * fermi_f
	f_integrand_den = den_o_s_pos * fermi_f
		
	tau_cum = integrate.cumtrapz(f_integrand_num,energies_pos,initial=0.) / integrate.cumtrapz(f_integrand_den,energies_pos,initial=0.1)
	
	return tau_cum 

# compute tau(T) at fermi level. value is energy-averaged. only positive energies are counted. 
def tau_cum_T_mp(temps,k_pots,ene,dos,dop,rep,que):
	"""
	same as tau_cum_T but with multiprocessing.
	compute mean relaxation time across all energies. it is given by (5.27) of fund of semiconductors yu,cardona 4th ed. use SI UNITS!!
	input:
	- 1 temperature
	- 1 fermi energy
	- array of energies
	- array of dos
	- array of dopings
	- number of replicae for phonon
	output:
	- array of cumulative tau_ave_T  i.e. integral of tau up to energy e_pos
	"""
	
	# get only the positive energies, and associated dos, and fermi function values, and tau_T(E) values
	
	energies_pos_indx = [ene > 0]
	energies_pos = ene[tuple(energies_pos_indx)]
	den_o_s_pos = dos[tuple(energies_pos_indx)]
	fermi_f = np.zeros_like(energies_pos)
	tau_T = np.zeros_like(energies_pos)
	dfde = np.zeros_like(energies_pos)
	tau_cum = np.zeros((len(temps),len(k_pots),len(energies_pos)))
	
	
	for i_T,val_T in enumerate(temps):
		for i_kp,v_kp in enumerate(k_pots):
			for i_en,val_en in enumerate(energies_pos):
				fermi_f[i_en] = fermi(val_en,val_T,v_kp)
				if dop[i_T][i_kp] < 0:
					dop[i_T][i_kp] = -dop[i_T][i_kp]
				tau_T[i_en] = tau_tot(val_T,v_kp,val_en,rep,dop[i_T][i_kp])
			
			dfde = fd.dFDde(energies_pos,v_kp,units.BOLTZMANN_SI*val_T)
		
			f_integrand_num = 2./3. * tau_T * energies_pos * den_o_s_pos * (-dfde)
			#f_integrand_num = tau_T * den_o_s_pos * fermi_f
			f_integrand_den = den_o_s_pos * fermi_f
			
			tau_cum[i_T][i_kp] = integrate.cumtrapz(f_integrand_num,energies_pos,initial=0.) / integrate.cumtrapz(f_integrand_den,energies_pos,initial=1)
	que.put(tau_cum)

def tau_ave(Temps,mu,ene,dos):
	"""
	compute mean relaxation time across all energies. For all temperatures and chemical potentials in input array Tr,mu. it is given by integral like (5.27) of fund of semiconductors yu,cardona 4th ed. only for positive energies. use SI UNITS!!
	input:
		- array of temperature
		- array of chemical potential
		- array of energies
		- array of dos
	output:
		- tau_ave(T,mu) (array of dimensions Temps x mu )
	"""
	tau_avg = np.zeros((len(Temps),len(mu)))
	for i_T,val_T in enumerate(Temps):
		for i_mu,val_mu in enumerate(mu):
			tau_avg[i_T][i_mu] = tau_ave_T(val_T,val_mu,ene,dos)[len(tau_avg)-1]
	
	return tau_avg


def positive_energies(ene):
	"""
	given an array of energies returns an array of only the positive ones
	input: energies (array)
	output: positive energies (array)
	"""
	energies_pos_indx = [ene > 0]
	energies_pos = ene[tuple(energies_pos_indx)]
	return energies_pos







































































