import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIRDModel(beta, gamma, δ1, δ2,tend,vaccinateAfter,minI,maxI,distanceModel):
    # Total population, N.
    N = 1.0
    # Initial number of infected and recovered individuals, I0 and R0.
    S0 = 0.99999
    I0 = 0.00001
    R0 = 0.0
    D0 = N - S0 - I0 - R0
    t = np.linspace(0.0, tend, int(tend))
    # The SIRD model differential equations.
    def rhs(SIRD, t, N, beta, gamma, δ1, δ2, vaccinateAfter, minI,maxI):
        S, I, R, D = SIRD
        if (distanceModel == 'Reactive'):
            δf2 = 0. #δ2
            δf1 = 0. #δ1

            if (I>=maxI):
                δf1 = 0.0
                δf2 = δ2 #* np.sin(2.0*np.pi*t/200)**2

            if (I <= minI):
                δf1 = δ1 #- beta/N * I #* np.sin(2.0*np.pi*t/200)**2
                δf2 = 0.0
        else:
            δf2 = δ2 
            δf1 = δ1             

        vacc = 0.0
        if (t>= vaccinateAfter):
            vacc = 1.0
            δf2 = 0.0 #* np.sin(2.0*np.pi*t/20)
            δf1 = 1.0 #* np.sin(2.0*np.pi*t/200)**2
            beta = 0.0

        dSdt = - beta/N * S * I + δf1*D - δf2*S - vacc*S 
        dIdt =   beta/N * S * I - gamma * I
        dRdt =   gamma * I +  vacc*S 
        dDdt = - δf1*D + δf2*S
        return dSdt, dIdt, dRdt, dDdt

    # Initial conditions
    y0 = S0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    sol = odeint(rhs, y0, t, args=(N, beta, gamma, δ1, δ2,vaccinateAfter, minI, maxI))
    return t, sol.T

def plot_sird_model(infection_rate, incubation_period, D_to_S, S_to_D, tend_months, vaccinateAfter, minIpercent=1,maxIpercent=10,distanceModel='Constant',semilogy=False):
    tend = 30.0*tend_months
    gamma = 1.0/incubation_period
    vaccinateAfter = 30*vaccinateAfter
    δ1 = D_to_S
    δ2 = S_to_D
    
    t,sol = SIRDModel(infection_rate, gamma, δ1, δ2, tend, vaccinateAfter,minIpercent/100.,maxIpercent/100., distanceModel)
    S, I, R, D = sol
    plt.figure(figsize=[7,4])
#     plt.clf()   
    f = plt.plot
    if semilogy:
        f = plt.semilogy
    f(t/30, S*100, 'b', alpha=0.65, lw=2, label='Susceptible')
    f(t/30, R*100, 'g', alpha=0.65, lw=2, label='Recovered')    
    f(t/30, I*100, 'r', alpha=0.65, lw=2, label='Infected')
    f(t/30, D*100, 'k', alpha=0.65, lw=2, label='Distanced')

    # plot maximum infected line
    maxInfected = np.max(I)*100
    plt.axhline(y=maxInfected,xmin=0,xmax=tend,linestyle='-.',color='r')

    # plot minimum infected line
    maxQuarantined = np.max(D)*100
    plt.axhline(y=maxQuarantined,xmin=0,xmax=tend,linestyle='-.',color='k')

    # fix the yticks
    if δ2 > 0.0 and semilogy != True:
        ticks = [0.0,50,100,maxInfected, maxQuarantined]
        plt.yticks(ticks)

    plt.xlabel('# Months since outbreak')
    plt.ylabel('% Population')
    plt.xlim(0,tend/30)
    plt.yticks()
    plt.minorticks_on()    
#     plt.ylim(top=100)
    plt.legend()
    title = 'SIRD with $\delta_1$ =' + str(δ1) + ' $\delta_2$ = ' + str(δ2)
    plt.title(title)
    plt.grid()
#     plt.savefig(title + '.pdf')    