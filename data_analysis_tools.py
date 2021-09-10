import numpy as np
from numpy import r_
from matplotlib import pyplot as plt
import glob
import pandas as pd
import seaborn as sns
import scipy
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from sklearn.metrics import r2_score
from scipy.stats.distributions import t, norm

def pcov_to_sigma(pcov,*args,conf=0.682689492137086):
    '''Following a fit with curve_fit, returns sigmas from covariance.
    Default is one standard deviation with 68.2689% confidence.
    Set conf to 0.95 for 95% confidence interval (2 standard deviations).

    Assumes a normal distribution, unless provided with data and pars
    in which case a t-distribution is used'''
    if len(args)==0:
        tval = norm.ppf((1+conf)/2)
    elif len(args)==2:
        a = len(args[0])
        b = len(args[1])
        dof = abs(a-b)
        tval = t.ppf((1+conf)/2, dof)
    else:
        raise TypeError('pcov_to_sigma takes either one argument (pcov), or three arguments (pcov,data,pars), along with an optional named argument (conf)')

    sigmas = np.sqrt(np.diag(pcov))*tval
    return sigmas

def T2_strech_fit(x,a,b,c,noise_floor):
    return np.sqrt((a*np.exp(-(x/b)**c))**2+noise_floor**2)
def T2_fit(x,a,b,noise_floor):
    return np.sqrt((a*np.exp(-(x/b)))**2+noise_floor**2)
def T2_biexp_fit(x,a,b,c,d,noise_floor):
    return np.sqrt((a*np.exp(-x/b)+c*np.exp(-x/d))**2+noise_floor**2)

def T1_fit(x,a,b):
    return a*(1-np.exp(-x/b))

def T1_inversion_fit(x,a,b,c):
    return a*np.exp(-(x/b))+c

def T1_biexponential_fit(x,a,b,c):
    return a*np.exp(-(x/b))+c*np.exp(-(x/d)) + e





class T2(object):
    """Analysis for T2 measurements - fitting and plotting
       takes raw waits, I and Q"""

    def __init__(self,waits,Is,Qs,int_lims,T2_type="strech"):
        self.waits = waits
        self.Is = Is
        self.Qs = Qs
        self.lims = int_lims
        self.single_shot = len(np.shape(self.Is)) != 3
        self.T2_type = T2_type

    def return_ints(self,average_type='mean',return_values=True,return_quadrature_ints=False,save_to_file=False,folder="",fname=""):
        """Returns echo amplitude using mag averaging
           average_type: mean or max
           return_quadrature_ints: Bool, returns [mag_int,I_int,Q_int]
           save_to_file: Bool
           folder: str
           fname: str"""

        start,stop = self.lims
        if self.single_shot == True:
            print("\nsingle shot mode")
            I_ints,Q_ints,mag_ints = [],[],[]
            for idx in range(len(self.waits)):
                # Background Subtract on echo - single shot
                I = self.Is[idx] - np.mean(self.Is[idx][350:])
                Q = self.Qs[idx] - np.mean(self.Qs[idx][350:])

                mag = np.sqrt(np.square(I)+np.square(Q))
                mag_ints.append(np.trapz(mag[start:stop]))
                I_ints.append(np.mean(I[start:stop]))
                Q_ints.append(np.mean(Q[start:stop]))
            if return_values == False:
                self.mag_ints = mag_ints
                return 0
            elif return_quadrature_ints == True:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",[mag_ints,I_ints,Q_ints])
                self.mag_ints = mag_ints
                return [mag_ints,I_ints,Q_ints]
            else:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",mag_ints)
                self.mag_ints = mag_ints
                return mag_ints

        else:
            print("\nCalculating echos with {} averages".format(np.shape(self.Is)[1]))
            I_ints,Q_ints,mag_ints_mean,mag_ints = [],[],[],[]

            for idx in range(len(self.waits)):
                # Background Subtract echo for every average
                I = [self.Is[idx][j] - np.mean(self.Is[idx][j][300:])\
                                                for j in range(len(self.Is[idx]))]
                Q = [self.Qs[idx][j] - np.mean(self.Qs[idx][j][300:])\
                                                for j in range(len(self.Qs[idx]))]

                mag = np.sqrt(np.square(I)+np.square(Q))
                if average_type == "max":
                    mag_ints.append(np.max(np.mean([mag[j][start:stop] for j in range(len(mag))],1)))
                    I_ints.append(np.max(np.mean([I[j][start:stop] for j in range(len(mag))],1)))
                    Q_ints.append(np.max(np.mean([Q[j][start:stop] for j in range(len(mag))],1)))
                elif average_type == "mean":
                    mag_ints.append(np.mean(np.mean([mag[j][start:stop] for j in range(len(mag))],1)))
                    I_ints.append(np.mean(np.mean([I[j][start:stop] for j in range(len(mag))],1)))
                    Q_ints.append(np.mean(np.mean([Q[j][start:stop] for j in range(len(mag))],1)))
                elif average_type == "point":
                    mean_mag = np.mean(mag,0)
                    mag_ints.append(np.mean(mean_mag[start:stop]))
                elif average_type == None:
                    mag_ints.append([(np.mean([mag[j][start:stop] for j in range(len(mag))],1))])
                    I_ints.append(np.mean([I[j][start:stop] for j in range(len(mag))],1))
                    Q_ints.append(np.mean([Q[j][start:stop] for j in range(len(mag))],1))
                else:
                    raise Exception("Please choose valid averaging option: mean,max,None")

            if return_values == False:
                self.mag_ints = mag_ints
                return 0
            if return_quadrature_ints == True:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",[mag_ints,I_ints,Q_ints])
                self.mag_ints = mag_ints
                return [mag_ints,I_ints,Q_ints]
            else:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",mag_ints)
                self.mag_ints = mag_ints
                return mag_ints


    def fit(self,noise_floor_lim=-5,print_T2=True,save=False,folder="",fname="",plot=True,show=True,title=""):
        """Fits T2, returns [amp,T2,stretch_param] and errors"""
        self.noise_floor_lim = noise_floor_lim # determines start of noise floor average
        self.noise_floor = np.mean(self.mag_ints[self.noise_floor_lim:])

        if self.T2_type == "strech":
            popt,pcov = curve_fit(lambda x,a,b,c: T2_strech_fit(x,a,b,c,self.noise_floor),2*self.waits,self.mag_ints,bounds = ([0,0,1],[np.inf,np.inf,np.inf]))
        if self.T2_type == "normal":
            popt,pcov = curve_fit(lambda x,a,b: T2_fit(x,a,b,self.noise_floor),2*self.waits,self.mag_ints,bounds = ([0,0],[np.inf,np.inf]))
        if self.T2_type == "biexponential":
            popt,pcov = curve_fit(lambda x,a,b,c,d: T2_biexp_fit(x,a,b,c,d,self.noise_floor),2*self.waits,self.mag_ints,bounds = ([0,0,0,0],[np.inf,np.inf],np.inf,np.inf))
        sigmas = pcov_to_sigma(pcov,self.mag_ints,popt)
        self.popt = popt
        self.sigmas = sigmas
        if print_T2 == True: print(u"\nT2 = %.3f $\pm$ %.3f \u03bcs"%(popt[1]*1e6,sigmas[1]*1e6))
        if plot == True:
            self.plot(title)
            if save == True: plt.savefig(filename)
            if show == True: plt.show()

        return popt,sigmas

    def plot(self,title=""):

        plt.plot(self.waits*1e6,self.mag_ints,'o')
        if self.T2_type == "strech":
            plt.plot(self.waits*1e6,T2_strech_fit(2*self.waits,*self.popt,self.noise_floor),label = u"\nT2 = %.1f $\pm$ %.1f \u03bcs"%(self.popt[1]*1e6,self.sigmas[1]*1e6))
        if self.T2_type == "normal":
            plt.plot(self.waits*1e6,T2_fit(2*self.waits,*self.popt,self.noise_floor),label = u"\nT2 = %.1f $\pm$ %.1f \u03bcs"%(self.popt[1]*1e6,self.sigmas[1]*1e6))
        plt.title(title)
        plt.xlabel(u"wait (\u03bcs)")
        plt.ylabel("Integrated signal (V)")
        plt.tight_layout()
        plt.xlim(0,1e6*self.waits.max())
        plt.ylim(0)
        plt.legend()

    def analyse(self):
        self.return_ints(return_values=False)
        popt, sigmas = self.fit()
        return popt,sigmas

class EDFS(object):
    """Analysis for EDFS - plotting"""

    def __init__(self,fields,Is,Qs,int_lims):
        self.fields = fields*1e3
        self.Is = Is
        self.Qs = Qs
        self.lims = int_lims

    def integrate_echos(self,return_values=True,return_quadrature_ints=False):
        start,stop = self.lims
        I_ints,Q_ints,mag_ints = [],[],[]
        for idx in range(len(self.fields)):
            # Background Subtract on echo - single shot
            I = self.Is[idx] - np.mean(self.Is[idx][300:])
            Q = self.Qs[idx] - np.mean(self.Qs[idx][300:])

            mag = np.sqrt(np.square(I)+np.square(Q))
            mag_ints.append(np.mean(mag[start:stop]))
            I_ints.append(np.mean(I[start:stop]))
            Q_ints.append(np.mean(Q[start:stop]))

        self.mag_ints = mag_ints
        self.I_ints = I_ints
        self.Q_ints = Q_ints

        if return_values == False:
            return 0
        elif return_quadrature_ints == False:
            return mag_ints
        elif return_quadrature_ints == True:
            return [mag_ints,I_ints,Q_ints]

    def plot(self,plot_cw=False,save=False,folder="",fname="",show=True,title=""):
        self.integrate_echos(return_values=False)
        plt.plot(self.fields,self.mag_ints,'-+',lw=.5)
        plt.xlim(np.min(self.fields),np.max(self.fields))
        plt.ylim(0)
        plt.xlabel("Field (mT)")
        plt.ylabel("Echo Amplitude (a.u)")

        if show == True: plt.show()

class three_pulse_echo(object):
    """docstring for three_pulse_echo."""

    def __init__(self,waits,Is,Qs,taus,int_lims,avgs):
        self.waits = waits
        self.Is = Is
        self.Qs  = Qs
        self.taus = taus
        self.lims = int_lims
        self.avgs = avgs
        if self.avgs == 1:
            self.single_shot = True
        else:
            self.single_shot = False

    def integrate_echos(self,return_values=True,return_quadrature_ints=False):
        start,stop = self.lims
        I_ints_array,Q_ints_array,mag_ints_array = [],[],[]
        if self.single_shot == True:
            for i in range(len(self.taus)):
                I_ints,Q_ints,mag_ints = [],[],[]
                for idx in range(len(self.waits)):
                    # Background Subtract on echo - single shot
                    I = self.Is[i][idx][0] - np.mean(self.Is[-1][-1][0][300:])
                    Q = self.Qs[i][idx][0] - np.mean(self.Qs[-1][-1][0][300:])

                    mag = np.sqrt(np.square(I)+np.square(Q))
                    mag_ints.append(np.mean(mag[start:stop]))
                    I_ints.append(np.mean(I[start:stop]))
                    Q_ints.append(np.mean(Q[start:stop]))
                I_ints_array.append(I_ints)
                Q_ints_array.append(Q_ints)
                mag_ints_array.append(mag_ints)
            self.mag_ints = mag_ints_array
            self.I_ints = I_ints_array
            self.Q_ints = Q_ints_array
        else:
            for i in range(len(self.taus)):
                I_ints,Q_ints,mag_ints = [],[],[]
                for idx in range(len(self.waits)):
                    mags,Is,Qs = [],[],[]
                    for avg in range(self.avgs):
                        # Background Subtract on echo - single shot
                        I = self.Is[i][idx][avg] - np.mean(self.Is[-1][-1][avg][300:])
                        Q = self.Qs[i][idx][avg] - np.mean(self.Qs[-1][-1][avg][300:])

                        mag = np.sqrt(np.square(I)+np.square(Q))

                        mags.append(np.mean(mag[start:stop]))
                        Is.append(np.mean(I[start:stop]))
                        Qs.append(np.mean(Q[start:stop]))

                    I_ints.append(np.mean(Is))
                    Q_ints.append(np.mean(Qs))
                    mag_ints.append(np.mean(mags))

                I_ints_array.append(I_ints)
                Q_ints_array.append(Q_ints)
                mag_ints_array.append(mag_ints)
            self.mag_ints = mag_ints_array
            self.I_ints = I_ints_array
            self.Q_ints = Q_ints_array

        if return_values == False:
            return 0
        elif return_quadrature_ints == False:
            return mag_ints
        elif return_quadrature_ints == True:
            return [mag_ints,I_ints,Q_ints]

    def gamma_eff(self,waits,gamma_0,x,R,tau):
        return gamma_0 + 0.5*x * (R*tau + 1 - np.exp(-R*waits))

    def SD_amp(self,waits,gamma_0,x,R,T1,tau):

        return np.exp(-(waits/T1 + 2 * np.pi * tau * self.gamma_eff(waits,gamma_0,x,R,tau)))

    def add_noise_floor(self,data,noise):
        return np.abs(np.array(data) + 1j*noise) # Only works if data is not complex

    def SD_amp_concat(self,waits,gamma_0,x,R,noise,T1):
        #    (x,y) = xy_mesh

        #    gamma_0 = 0.8e3 # 3kHz for Hee-Jin
        #    gamma_sd = 300e3
        #    R = 0.010e3
        #    T1 = 0.696
        self.noise_floor = noise

        #noise1 = self.mag_ints[0][-1]
        #noise2 = self.mag_ints[1][-1]
        #noise3 = self.mag_ints[2][-1]

        amp = np.mean(self.mag_ints[0][0])
        #relative_amp = 0.10
        #noise_floor = 0.0176
        #noise_floor = noise

        tau = self.taus[0]
        predicted_amp_30 = self.SD_amp(waits,gamma_0,x,R,T1,tau)
        tau = self.taus[1]
        predicted_amp_50 = self.SD_amp(waits,gamma_0,x,R,T1,tau)
        tau = self.taus[2]
        predicted_amp_80 = self.SD_amp(waits,gamma_0,x,R,T1,tau)

        amp_30 = self.add_noise_floor(amp*predicted_amp_30,noise)
        amp_50 = self.add_noise_floor(amp*predicted_amp_50,noise)
        amp_80 = self.add_noise_floor(amp*predicted_amp_80,noise)

        return r_[amp_30,amp_50,amp_80].squeeze()

    def fit(self):
        self.integrate_echos(return_values=False)

        data = np.array(self.mag_ints).flatten()
        #self.noise_floor = np.mean(self.mag_ints[-1][-10:])
        waits = self.waits
        gamma_0 = 0.8e3
        x = 40e3
        R = 60
        T1 = 1
        amp = 0.09
        #sigma = np.tile(r_[np.repeat([np.inf],40),np.repeat([1],140),np.repeat([1],21)],3)
        par_names = ['gamma_0','x','R','noise','T1']
        #initial_guess = [gamma_0,x,R,amp]
        bounds = ([0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf])

        pars, pcov = curve_fit(lambda waits,gamma_0,x,R,noise,T1: self.SD_amp_concat(waits,gamma_0,x,R,noise,T1), waits, data, bounds=bounds)#, max_nfev=1000)#, sigma=sigma)
        sigmas = pcov_to_sigma(pcov,data,pars)
        model_data = self.SD_amp_concat(waits,*pars)

        gamma_0 = pars[0]
        gamma_sd = pars[1]#*1e-3
        R = pars[2]
        #amp = pars[3]



        for i in range(len(pars)):
            print('{:>8}: {:.6f} ± {:.6f}'.format(par_names[i],pars[i],sigmas[i]))

        print('\nRsquare = {}'.format(r2_score(data[40:],model_data[40:])))

        model_data[len(self.Is[0])-1] = np.nan; model_data[2*len(self.Is[0])-1] = np.nan; model_data[3*len(self.Is[0])-1] = np.nan;
        data[len(self.Is[0])-1] = np.nan; data[2*len(self.Is[0])-1] = np.nan; data[3*len(self.Is[0])-1] = np.nan;


        plt.rcParams.update({'font.size': 8})
        #sns.set(style="ticks", palette="deep", color_codes=True,font_scale=3)
        plt.figure(figsize=(5,3.5))

        plt.plot(waits*1e3,model_data[0:len(self.Is[0])],"tab:blue",lw=1)
        plt.plot(waits*1e3,model_data[len(self.Is[0]):2*len(self.Is[0])],"tab:orange",lw=1)
        plt.plot(waits*1e3,model_data[2*len(self.Is[0]):3*len(self.Is[0])],"tab:green",lw=1)
        plt.plot(waits*1e3,data[0:len(self.Is[0])],"o",ms=2,alpha=0.5)
        plt.plot(waits*1e3,data[len(self.Is[0]):2*len(self.Is[0])],"o",ms=2,alpha=0.5)
        plt.plot(waits*1e3,data[2*len(self.Is[0]):3*len(self.Is[0])],"o",ms=2,alpha=0.5)
        #plt.text(0.75,0.060,'T$_1$ = {:.0f} ms (fixed)\n$\Gamma_0$ = {:.0f} $\pm$ {:.0f} Hz\n$\gamma_{{Y}}$ = {:.2f} $\pm$ {:.2f} MHz/T\nR = {:.0f} $\pm$ {:.0f} Hz'.format(T1,gamma_0,sigmas[0],gamma_sd,sigmas[1]*1e-6,R,sigmas[2]),bbox=dict(facecolor='white'))
        plt.legend([r'$\tau$ = {:3.0f} $\mu$s'.format(i) for i in np.multiply(self.taus,1e6)])
        #plt.xlim(0,20)
        #plt.ylim(0,0.08)
        plt.xlabel("T$_w$ (ms)")
        plt.ylabel("Echo Amplitude (arb. u)")
        #plt.show()
        #plt.tight_layout()
        #plt.savefig("new_fig4.pdf")
        return pars,sigmas

class T1(object):
    """docstring for T1."""

    def __init__(self,Is,Qs,long_waits,int_lims,type='saturation'):
        self.Is = Is
        self.Qs = Qs
        self.long_waits = long_waits
        self.lims = int_lims
        self.single_shot = len(np.shape(self.Is)) != 3
        self.type = type

    def return_ints(self,average_type='mean',return_values=True,return_quadrature_ints=False,save_to_file=False,folder="",fname=""):
        """Returns echo amplitude using mag averaging
           lims: [start,stop] for echo integration
           average_type: mean or max
           return_quadrature_ints: Bool, returns [mag_int,I_int,Q_int]
           save_to_file: Bool
           folder: str
           fname: str"""

        start,stop = self.lims
        if self.single_shot == True:
            print("\nsingle shot mode")
            I_ints,Q_ints,mag_ints = [],[],[]
            for idx in range(len(self.long_waits)):
                # Background Subtract on echo - single shot
                I = self.Is[idx] - np.mean(self.Is[idx][300:])
                Q = self.Qs[idx] - np.mean(self.Qs[idx][300:])

                mag = np.sqrt(np.square(I)+np.square(Q))
                mag_ints.append(np.mean(mag[start:stop]))
                I_ints.append(np.mean(I[start:stop]))
                Q_ints.append(np.mean(Q[start:stop]))
            if return_values == False:
                self.mag_ints = mag_ints
                return 0
            elif return_quadrature_ints == True:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",[mag_ints,I_ints,Q_ints])
                self.mag_ints = mag_ints
                self.I_ints = I_ints
                self.Q_ints = Q_ints
                return [mag_ints,I_ints,Q_ints]
            else:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",mag_ints)
                self.mag_ints = mag_ints
                return mag_ints

        else:
            print("\nCalculating echos with {} averages".format(np.shape(self.Is)[1]))
            I_ints,Q_ints,mag_ints_mean,mag_ints = [],[],[],[]

            for idx in range(len(self.long_waits)):
                # Background Subtract echo for every average
                I = [self.Is[idx][j] - np.mean(self.Is[idx][j][300:])\
                                                for j in range(len(self.Is[idx]))]
                Q = [self.Qs[idx][j] - np.mean(self.Qs[idx][j][300:])\
                                                for j in range(len(self.Qs[idx]))]

                mag = np.sqrt(np.square(I)+np.square(Q))
                if average_type == "max":
                    mag_ints.append(np.max(np.mean([mag[j][start:stop] for j in range(len(mag))],1)))
                    I_ints.append(np.max(np.mean([I[j][start:stop] for j in range(len(mag))],1)))
                    Q_ints.append(np.max(np.mean([Q[j][start:stop] for j in range(len(mag))],1)))
                elif average_type == "mean":
                    mag_ints.append(np.mean(np.mean([mag[j][start:stop] for j in range(len(mag))],1)))
                    I_ints.append(np.mean(np.mean([I[j][start:stop] for j in range(len(mag))],1)))
                    Q_ints.append(np.mean(np.mean([Q[j][start:stop] for j in range(len(mag))],1)))
                elif average_type == None:
                    mag_ints.append([(np.mean([mag[j][start:stop] for j in range(len(mag))],1))])
                    I_ints.append(np.mean([I[j][start:stop] for j in range(len(mag))],1))
                    Q_ints.append(np.mean([Q[j][start:stop] for j in range(len(mag))],1))
                else:
                    raise Exception("Please choose valid averaging option: mean,max,None")

            if return_values == False:
                self.mag_ints = mag_ints
                return 0
            if return_quadrature_ints == True:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",[mag_ints,I_ints,Q_ints])
                self.mag_ints = mag_ints
                return [mag_ints,I_ints,Q_ints]
            else:
                if save_to_file == True:
                    if fname[-4:] == ".txt":
                        fname = fname[:-4]
                    np.savetxt(folder+"\\"+fname+".txt",mag_ints)
                self.mag_ints = mag_ints
                return mag_ints

    def T1_fit(self,print_T2=True,save=False,folder="",fname="",plot=True,show=True,title=""):
        """Fits T2, returns [amp,T2,stretch_param] and errors"""
        if self.type == "saturation":
            self.noise_floor = np.mean(self.mag_ints[-5:])
            self.mag_ints = self.mag_ints - np.min(self.mag_ints)
            #def T2_stretchedfit(mag_int,t,noise_floor = 0,print_T2 = True,plot = True,title = None, show = True,save=False,filename = 'T2.pdf'):
            popt,pcov = curve_fit(lambda x,a,b: T1_fit(x,a,b),self.long_waits,self.mag_ints,bounds = ([0,0],[np.inf,np.inf]))
            sigmas = pcov_to_sigma(pcov,self.mag_ints,popt)

            if print_T2 == True: print(u"\nT1 = %.3f $\pm$ %.3f \u03bcs"%(popt[1],sigmas[1]))
            if plot == True:
                self.plot(popt,sigmas,title)
                if save == True: plt.savefig(filename)
                if show == True: plt.show()
            return popt,sigmas
        if self.type == "inversion":
            self.noise_floor = np.mean(self.I_ints[-3:])
            #def T2_stretchedfit(mag_int,t,noise_floor = 0,print_T2 = True,plot = True,title = None, show = True,save=False,filename = 'T2.pdf'):
            popt,pcov = curve_fit(lambda x,a,b,c: T1_inversion_fit(x,a,b,c),self.long_waits,self.I_ints,bounds = ([0,0,0],[np.inf,np.inf,np.inf]))
            sigmas = pcov_to_sigma(pcov,self.I_ints,popt)

            if print_T2 == True: print(u"\nT1 = %.3f $\pm$ %.3f \u03bcs"%(popt[1],sigmas[1]))
            if plot == True:
                self.plot(popt,sigmas,title)
                if save == True: plt.savefig(filename)
                if show == True: plt.show()
            return popt,sigmas

        if self.type == "biexponential":
            self.noise_floor = np.mean(self.I_ints[-3:])
            popt,pcov = curve_fit(lambda x,a,b,c,d,e: T1_biexponential_fit(x,a,b,c,d,e),self.long_waits,self.I_ints,bounds = ([0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf]))
            sigmas = pcov_to_sigma(pcov,self.I_ints,popt)

            if print_T2 == True: print(u"\nT1 = %.3f $\pm$ %.3f \u03bcs\nT1 = %.3f $\pm$ %.3f \u03bcs"%(popt[1],sigmas[1],popt[3],sigmas[3]))
            if plot == True:
                self.plot(popt,sigmas,title)
                if save == True: plt.savefig(filename)
                if show == True: plt.show()
            return popt,sigmas


    def plot(self,popt,sigmas,title):
        if self.type == "saturation":

            plt.plot(self.long_waits,self.mag_ints,'o')
            plt.plot(self.long_waits,T1_fit(self.long_waits,*popt),label = u"\nT1 = %.3f $\pm$ %.3f s"%(popt[1],sigmas[1]))
            plt.title(title)
            plt.xlabel(u"wait (s)")
            plt.ylabel("Integrated signal (V)")
            plt.tight_layout()
            plt.xlim(0,self.long_waits.max())
            plt.ylim(0)
            plt.legend()

        if self.type == "inversion":

            plt.plot(self.long_waits,self.I_ints,'o')
            plt.plot(self.long_waits,T1_inversion_fit(self.long_waits,*popt),label = u"\nT1 = %.3f $\pm$ %.3f s"%(popt[1],sigmas[1]))
            plt.title(title)
            plt.xlabel(u"wait (s)")
            plt.ylabel("Integrated signal (V)")
            plt.tight_layout()
            plt.xlim(0,self.long_waits.max())
            plt.legend()

        if self.type == "biexponential":
            plt.plot(self.long_waits,self.I_ints,'o')
            plt.plot(self.long_waits,T1_biexponential_fit(self.long_waits,*popt),label = u"\nT1 = %.3f $\pm$ %.3f s\n\nT1 = %.3f $\pm$ %.3f s"%(popt[1],sigmas[1],popt[3],sigmas[3]))
            plt.title(title)
            plt.xlabel(u"wait (s)")
            plt.ylabel("Integrated signal (V)")
            plt.tight_layout()
            plt.xlim(0,self.long_waits.max())
            plt.legend()

class T2_temp_dep(object):
      """docstring for T2_temp_dep."""

      def __init__(self, T2s,T2_errs, temps, higher_levels, lower_levels,freqs,amps,populations,n_species,abundance,g_A,field,calc_res = True,density=1):
          self.T2s = T2s
          self.T2_errs = T2_errs
          self.temps = temps
          self.higher_lvls = higher_levels
          self.lower_lvls = lower_levels
          self.gamma = np.divide(1,T2s)
          self.gamma_errs = T2_errs*np.divide(1,np.square(T2s))
          self.freqs = freqs
          self.pops = populations
          self.n_species = n_species
          self.abundance = abundance
          self.n_sites = len(self.freqs)/len(self.abundance)
          self.amps = amps
          self.g_A = g_A
          self.field = field
          self.calc_res = calc_res
          self.density = density

      def get_Z_is(self,freqs):
          T_zs  = []
          for freq in freqs:
              T_z = scipy.constants.Planck*freq/scipy.constants.Boltzmann
              T_zs.append(T_z)
          return T_zs

      def Gamma_fit(self,T,dipole,gamma_0):
          sum_tot = 0
          Ts = np.linspace(10,1.2,100)
          highers = self.higher_lvls
          lowers = self.lower_lvls
          Tz_array = []
          isotope = 0
          sum_tot_array = []
          print(len(T))
          test_Ts = self.temps
          amp_Ts_idx = [np.argmin(abs(T[i]-Ts)) for i in range(len(T))]
          print(amp_Ts_idx)
          T=np.array(T)
          for idx in range(self.n_species):
              print(idx)
              T_zs = self.get_Z_is(self.freqs[idx])
              Tz_array.append(T_zs)
              populations_array =[] #np.zeros((len(highers[isotope]),len(T)))
              #for i in range(len(highers[isotope])):
            # #     populations = []
                #  for j in range(len(T)):
            #          x = np.argmin(abs(np.subtract(Ts,T[j])))
            #          population = ((self.pops[isotope][highers[isotope][i]][x]))
            #          populations.append(population)
            #      populations_array.append(populations)

              for i in range(len(self.freqs[idx])):
                  c = (scipy.constants.physical_constants['Bohr magneton'][0])/scipy.constants.Planck/1e9
                  x = [np.argmin(abs(np.subtract(Ts,np.array(T)[j]))) for j in range(len(T))]
                  #print(idx,isotope,i)
                  #population = ((self.pops[isotope][highers[isotope][i]][x]))
                  sum_tot_a = []
                  #for iT in range(len(T)):

                  pol = 1/((1 + np.exp(T_zs[i]/T))*(1 + np.exp(-T_zs[i]/T)))
                  pop = np.exp(T_zs[i]/(2*T)) # higher level
                  #plt.plot(T,pol)
                  #plt.show()
                  #print(isotope)
                  #print(i)

                  print(len(self.amps[idx][i][amp_Ts_idx]))
                  print(len(2*T))
                  print(len( (2/(np.exp(T_zs[i]/(2*T))+np.exp(-T_zs[i]/(2*T))))))
                  print(T_zs[i])
                  print(dipole)
                  n_B = (self.density*(self.abundance[isotope]/self.n_sites))*np.exp(T_zs[i]/(2*T))
                  constants = (np.pi*scipy.constants.mu_0*(scipy.constants.physical_constants['Bohr magneton'][0]**2)/(9*np.sqrt(3)*scipy.constants.Planck))*self.g_A**2
                  sum_tot_i = ((np.sqrt(np.pi*self.amps[idx][i][amp_Ts_idx]*n_B*constants)/2) * (2/(np.exp(T_zs[i]/(2*T))+np.exp(-T_zs[i]/(2*T)))))#*(pop*self.density*(self.abundance[isotope]/self.n_sites))**(1/2)
                 #sum_tot_a.append(sum_tot_i)
                      #sum_tot += pop*((dipole*self.amps[idx][i]*pop))*(self.density*(self.abundance[isotope]/self.n_sites))**(3/2)#(pol)*(dipole)*(self.density*(self.abundance[isotope]/self.n_sites))**(3/2)#
                  sum_tot += sum_tot_i#np.array(sum_tot_a)
              if (idx+1)%self.n_sites == 0:
                  isotope+=1
          self.Tz_array = Tz_array

          #C = np.sqrt(np.pi*scipy.constants.mu_0*(scipy.constants.physical_constants['Bohr magneton'][0]**2)/(9*np.sqrt(3)*scipy.constants.Planck))#np.sqrt((4*(np.pi**2)*scipy.constants.mu_0*((scipy.constants.physical_constants['Bohr magneton'][0])**4)*self.g_A)/(9*np.sqrt(3)*scipy.constants.Planck))
          #C = 1
          return sum_tot

      def R_gammasd_fit(self,T,y):

            sum_tot = 0
            Ts = np.linspace(10,1.2,100)
            highers = self.higher_lvls
            lowers = self.lower_lvls
            Tz_array = []
            isotope = 0
            sum_tot_array = []
            print(len(T))
            test_Ts = self.temps
            amp_Ts_idx = [np.argmin(abs(T[i]-Ts)) for i in range(len(T))]
            print(amp_Ts_idx)
            T=np.array(T)
            for idx in range(self.n_species):
                print(idx)
                T_zs = self.get_Z_is(self.freqs[idx])
                Tz_array.append(T_zs)
                populations_array =[] #np.zeros((len(highers[isotope]),len(T)))
                #for i in range(len(highers[isotope])):
              # #     populations = []
                  #  for j in range(len(T)):
              #          x = np.argmin(abs(np.subtract(Ts,T[j])))
              #          population = ((self.pops[isotope][highers[isotope][i]][x]))
              #          populations.append(population)
              #      populations_array.append(populations)

                for i in range(len(self.freqs[idx])):
                    c = (scipy.constants.physical_constants['Bohr magneton'][0])/scipy.constants.Planck/1e9
                    x = [np.argmin(abs(np.subtract(Ts,np.array(T)[j]))) for j in range(len(T))]
                    #print(idx,isotope,i)
                    #population = ((self.pops[isotope][highers[isotope][i]][x]))
                    sum_tot_a = []
                    #for iT in range(len(T)):

                    pol = 1/((1 + np.exp(T_zs[i]/T))*(1 + np.exp(-T_zs[i]/T)))
                    pop = np.exp(T_zs[i]/(2*T)) # higher level
                    #plt.plot(T,pol)
                    #plt.show()
                    #print(isotope)
                    #print(i)

                    print(len(self.amps[idx][i][amp_Ts_idx]))
                    print(len(2*T))
                    print(len( (2/(np.exp(T_zs[i]/(2*T))+np.exp(-T_zs[i]/(2*T))))))
                    print(T_zs[i])
                    #print(dipole)
                    bohr = scipy.constants.physical_constants['Bohr magneton'][0]
                    mu0 = scipy.constants.mu_0
                    h = scipy.constants.Planck
                    n_B = (self.density*(self.abundance[isotope]/self.n_sites))*np.exp(T_zs[i]/(2*T))
                    gamma_sd = ((np.pi/(9*np.sqrt(3)))*(n_B*mu0*self.g_A*self.g_A*(bohr**2))/h)*1/(np.cosh(T_zs[i]/(T))**2)
                    R = self.amps[idx][i][amp_Ts_idx]
                    sum_tot_i = 1/np.sqrt(np.pi*R*gamma_sd)
                        #sum_tot += pop*((dipole*self.amps[idx][i]*pop))*(self.density*(self.abundance[isotope]/self.n_sites))**(3/2)#(pol)*(dipole)*(self.density*(self.abundance[isotope]/self.n_sites))**(3/2)#
                    sum_tot += sum_tot_i#np.array(sum_tot_a)
                if (idx+1)%self.n_sites == 0:
                    isotope+=1
            self.Tz_array = Tz_array

            #C = np.sqrt(np.pi*scipy.constants.mu_0*(scipy.constants.physical_constants['Bohr magneton'][0]**2)/(9*np.sqrt(3)*scipy.constants.Planck))#np.sqrt((4*(np.pi**2)*scipy.constants.mu_0*((scipy.constants.physical_constants['Bohr magneton'][0])**4)*self.g_A)/(9*np.sqrt(3)*scipy.constants.Planck))
            #C = 1
            return sum_tot


      def Gamma_fit_no_res(self,T,dipole):
          sum_tot = 0
          Ts = np.linspace(0,1.2,1000)
          highers = self.higher_lvls
          lowers = self.lower_lvls
          Tz_array = []
          isotope = 0


          for idx in range(self.n_species):

              T_zs = self.get_Z_is(self.freqs[idx])
              Tz_array.append(T_zs)
              populations = np.zeros((len(highers[isotope]),len(T)))
              for i in range(len(highers[isotope])):
                  for j in range(len(T)):
                      x = np.argmin(abs(np.subtract(Ts,T[j])))
                      population = ((self.pops[isotope][highers[isotope][i]][x]))
                      populations[i,j] = population

              for i in range(len(self.freqs[idx])):
                  pop = 1/((1 + np.exp(T_zs[i]/T))*(1 + np.exp(-T_zs[i]/T)))
                  sum_tot += dipole*pop*self.amps[idx][i]*(self.abundance[isotope]/self.n_sites)
          if idx%self.n_sites == 0:
              isotope+=1
          self.Tz_array = Tz_array
          return 1/(445*1e-6)+sum_tot



      def fit(self,plot_lines=True,show_bounds = False):
          self.plot_lines = plot_lines
          self.show_bounds = show_bounds
          T = np.linspace(0, 1.2, 1000)
          if self.calc_res == True:
              popt, pcov = curve_fit(lambda x,y: self.R_gammasd_fit(x,y),self.temps,self.gamma)#,bounds=([0,0],[np.inf,np.inf]))
          elif self.calc_res == False:
              popt, pcov = curve_fit(lambda x,xi: self.Gamma_fit_no_res(x,xi),self.temps,self.gamma,bounds=([0],[np.inf]))

          sigmas = pcov_to_sigma(pcov,self.gamma,popt)
          plt.rcParams.update({'font.size': 16})
          color="tab:red"
          #plt.errorbar(self.temps,self.gamma*1e-3,xerr=0,yerr=self.gamma_errs*1e-3,color='grey',ls="",marker='o',ms=5)
          #plt.plot(np.linspace(0, 1.2, 1000), self.Gamma_fit(np.linspace(0, 1.2, 1000), *popt)*1e-3,color='black')
          colors = ["blue","red","yellow","green","purple","orange"]
          labels = ["171Yb","171Yb","173Yb","173Yb","I=0 Yb",'I=0 Yb']
          self.lines = []
          if self.plot_lines == True:
              for idx in range(self.n_species):
                  Tzs = self.Tz_array[idx]
                  if self.calc_res == True:
                      self.lines.append(plt.vlines(Tzs,self.R_gammasd_fit(Tzs, *popt)*1e-3 - 5,self.R_gammasd_fit(Tzs, *popt)*1e-3 + 5,linestyles = "dashed",color=colors[idx],label=labels[idx]))
                  elif self.calc_res == False:
                      self.lines.append(plt.vlines(Tzs,self.Gamma_fit_no_res(Tzs, *popt)*1e-3 - 5,self.Gamma_fit_no_res(Tzs, *popt)*1e-3 + 5,linestyles = "dashed",color=colors[idx],label=labels[idx]))
          plt.errorbar(self.temps,self.gamma*1e-3,xerr=0,yerr=self.gamma_errs*1e-3,color='grey',ls="",marker='o',ms=5)
          if self.calc_res == True:
              plt.plot(np.linspace(0, 1.2, 1000), self.R_gammasd_fit(np.linspace(0, 1.2, 1000), *popt)*1e-3,color='black')
              if self.show_bounds == True:
                  plt.fill_between(np.linspace(0, 1.2, 1000),self.R_gammasd_fit(np.linspace(0, 1.2, 1000), *(popt-sigmas))*1e-3,self.R_gammasd_fit(np.linspace(0, 1.2, 1000), *(popt+sigmas))*1e-3,alpha=0.5)
          elif self.calc_res == False:
              plt.plot(np.linspace(0, 1.2, 1000), self.Gamma_fit_no_res(np.linspace(0, 1.2, 1000), *popt)*1e-3,color='black')
              if self.show_bounds == True:
                  plt.fill_between(np.linspace(0, 1.2, 1000),self.Gamma_fit_no_res(np.linspace(0, 1.2, 1000), *(popt-sigmas))*1e-3,self.Gamma_fit_no_res(np.linspace(0, 1.2, 1000), *(popt+sigmas))*1e-3,alpha=0.5)
          #plt.xlim(0,1.2)
          #plt.ylim(0,50)
          plt.xlabel("Temperature (K)")
          plt.ylabel(r"Decoherence Rate (1/T$_{2}$) (kHz)")

          return popt,sigmas,self.lines

class DEER(object):
    """Analysis for double electron-electron resonance"""

    def __init__(self,Is,Qs,deer_freqs):
        self.Is = Is
        self.Qs = Qs
        self.deer_freqs = deer_freqs

    def integrate_echos(self,start=90,stop=120,noise_idx=-30,return_values=False):
        """inegrates echos and returns pandas array of data"""
        self.start = start
        self.stop = stop
        self.noise_idx = noise_idx
        I = np.mean(self.Is,axis=1) - np.mean(np.mean(self.Is,axis=1)[-1][self.noise_idx:])
        Q = np.mean(self.Qs,axis=1) - np.mean(np.mean(self.Qs,axis=1)[-1][self.noise_idx:])
        mag = np.sqrt(np.square(I)+np.square(Q))
        self.mags = mag
        mag_ints,I_ints,Q_ints = [],[],[]
        for i in range(len(mag)):
            I_int = np.mean(I[i][start:stop])
            Q_int = np.mean(Q[i][start:stop])
            mag_int = np.mean(mag[i][start:stop])
            I_ints.append(I_int)
            Q_ints.append(Q_int)
            mag_ints.append(mag_int)
        self.I_ints = I_ints
        self.Q_ints = Q_ints
        self.mag_ints = mag_ints

        collected_data = {"Frequency (MHz)":self.deer_freqs[:len(mag_ints)],
                          "I ints (mV)": np.array(I_ints)*1e3,
                          "Q ints (mV)": np.array(Q_ints)*1e3,
                          "mag ints (mV)": np.array(mag_ints)*1e3}
        data = pd.DataFrame(collected_data)
        self.data = data
        if return_values == True:
            return data

    def analyse(self,start=90,stop=120,noise_idx=-30,plot=True,plot_I=False,plot_Q=False,return_data=False):
        self.integrate_echos(start=90,stop=120,noise_idx=-30)

        if plot == True:
            plt.plot(self.deer_freqs[:len(self.mag_ints)],self.mag_ints,lw=0.1,label="Mag",color="k")
        if plot_I == True:
            plt.plot(self.deer_freqs[:len(self.mag_ints)],self.I_ints,lw=0.1,label="I",color="tab:red")
        if plot_Q == True:
            plt.plot(self.deer_freqs[:len(self.mag_ints)],self.Q_ints,lw=0.1,label="Q",color="tab:pink")
        if plot == True:
            plt.ylim(0)
            plt.xlim(np.min(self.deer_freqs),np.max(self.deer_freqs))
            plt.xlabel("DEER Pulse Frequency (MHz)")
            plt.ylabel("Echo Amplitude (V)")
            plt.legend()

        if return_data == True:
            return self.data
