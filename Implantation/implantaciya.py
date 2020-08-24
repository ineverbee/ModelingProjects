print('''
    ---------------------------------------------------
    Моделирование процесса имплантации фосфора в мишень
    из полупроводникового материала с параметрами, ввод
    имыми в консоли 
    --------------------------------- made by ineverbee
    ''')
print('''
    ###Установка зависимостей###
    ''')
import os
os.system('pip install -r requirements.txt')
import matplotlib.pyplot as plt
import math, time
import numpy as np
from numpy import exp, sqrt, pi, inf, log

# Константы
q = 1.6e-19
k = 1.38e-23 / q  # эВ/К
q2 = 14.4 # эВ*ангстрем

def Sn(E, *args):
    m_dn, m_dp, epsilon, z1, z2, m1, m2, N = (args[i] for i in range(8))
    return 8.462e-15*z1*z2*m1*Sne(E, *args)/(m1+m2)/(z1**0.23+z2**0.23)*N*1e-4*1e-3  # keV/mkm

def Sne(E, *args):
    m_dn, m_dp, epsilon, z1, z2, m1, m2, N = (args[i] for i in range(8))
    a = 0.8854*0.529/(z1**(2/3)+z2**(2/3))**0.5   # angstrem
    e = a*m2*E*1e3/(z1*z2*q2*(m1+m2))
    if e >= 10:
        return log(e)/2/e
    else:
        return log(1+1.1383*e)/2/(e+0.01321*e**0.21226+0.19593*sqrt(e))

def Se(E, *args):
    m_dn, m_dp, epsilon, z1, z2, m1, m2, N = (args[i] for i in range(8))
    a = 0.8854 * 0.529 / (z1 ** (2 / 3) + z2 ** (2 / 3)) ** 0.5  # angstrem
    k0 = z1**(1/6)*0.0793*sqrt(z1*z2)*(m1+m2)**1.5/(z1**(2/3)+z2**(2/3))**0.75/m1**1.5/m2**0.5
    Cr = 4*pi*a**2*1e-16*N*m1*m2/(m1+m2)**2
    Ce = a*m2/(z1*z2*q2*(m1+m2))
    K = k0*Cr/sqrt(Ce)
    return K*sqrt(E*1e3)*1e-3*1e-4

def RdR(E, *args):
    m_dn, m_dp, epsilon, z1, z2, m1, m2, N = (args[i] for i in range(8))
    dE = 0.1
    rp = 0
    drpl = 0
    ksi = 0
    for i in range(1, E*10):
        rp = rp + (1-m2/m1*Sn(i/10, *args)/(Sn(i/10, *args)+Se(i/10, *args))*dE/(i/10))*dE/(Sn(i/10, *args)+Se(i/10, *args))
        ksi = ksi + 2*rp/(Se(i/10, *args)+Sn(i/10, *args))*dE
        drpl = drpl + (ksi-2*drpl)*m2/m1*Sn(i/10, *args)/(Sn(i/10, *args)+Se(i/10, *args))*dE/(i/10)
    drp = sqrt(ksi-rp**2-drpl**2)
    return rp, drp

def gauss(x, Rp, dRp, Q):
    return Q/((2*pi)**0.5*dRp*1e-4)*exp(-(x - Rp)**2/(2*dRp**2))

class Graph:
    Y = []
    X = []

    def draw(self, start, end, x, C):
        for i in range(start, end):
            self.X.append(x[i])
            self.Y.append(C[i])

        plt.plot(self.X, self.Y, c='blue', label='C(x)')
        plt.xlabel('x, мкм')
        plt.ylabel('Концентрация С')
        plt.legend()
        plt.xlim(0, x[-1])
        plt.show()
        return C

Graph = Graph()

def implant(sc, Q, E):
    n = 1000
    xmax = float(input('Enter max x value (0.1-1.0), mkm - '))
    dx = xmax / n
    C = np.empty(n)
    x = np.empty(n)

    if sc == 'Ge':
        # Параметры германия легированного фосфором 
        m_dn = 0.56
        m_dp = 0.37
        epsilon = 16.3
        z1 = 15
        z2 = 32
        m1 = 30.97
        m2 = 72.59
        N = 4.4e22  # cm**-3 плотность атомов мишени
    elif sc == 'Si':
        # Параметры кремния легированного фосфором 
        m_dn = 1.08
        m_dp = 0.59
        epsilon = 11.7
        z1 = 15
        z2 = 14
        m1 = 30.97
        m2 = 28.086
        N = 5e22  # cm**-3 плотность атомов мишени

    sn = Sn(E, m_dn, m_dp, epsilon, z1, z2, m1, m2, N)
    se = Se(E, m_dn, m_dp, epsilon, z1, z2, m1, m2, N)
    print("Sn = %0.2f кэВ/мкм  " % sn, "Se = %0.2f кэВ/мкм" % se)
    rp, drp = RdR(E, m_dn, m_dp, epsilon, z1, z2, m1, m2, N)
    print("_________________________________________________________________________________________________")
    print("Rp = %.4f " % rp, "dRp = %.4f " % drp)
    print("_________________________________________________________________________________________________")
    
    for i in range(0, n):
        if i == 0:
            x[i] = 0
        else:
            x[i] = x[i-1] + dx
        C[i] = gauss(x[i], rp, drp, Q)
    C = Graph.draw(0, n, x, C)

print('''
    ###Введение параметров###
    ''')
implant(
    input('Enter semiconductor (Si, Ge) - '),
    float(input('Enter dose (1e13-1e15), cm^-2 - ')),
    int(input('Enter energy (50-130), keV - ')),
)
input('Press ENTER to exit')