print('''
    -----------------------------------------
    Моделирование процесса диффузии примеси в
    кремнии после имплантации с параметрами, 
    вводимыми в консоли 
    ----------------------- made by ineverbee
    ''')
print('''
    ###Установка зависимостей###
    ''')
import os
os.system('pip install -r requirements.txt')
import matplotlib.pyplot as plt
import time
import numpy as np
from math import erf, exp, sqrt, pi, inf
from scipy.optimize import fsolve
from scipy.integrate import quad, simps

# Константы
q = 1.6e-19
k = 1.38e-23 / q  # эВ/К

def printProgressBar (iteration, total, prefix = '', suffix = '', length = 100):
    fill = '█'
    decimals = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()


class Si:

    def Eg(self, temperature):
        return 1.17 - 4.73e-4 / (temperature + 636) * temperature ** 2

    def Nv(self, temperature):
        return 4.82e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def Nc(self, temperature):
        return 4.82e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * exp(
            -1 * self.Eg(temperature) / (2 * temperature * k))

    def Rp(self, E):
        return self.a1*E**self.a2 + self.a3

    def dRp(self, E):
        return self.a4*E**self.a5 + self.a6

    def gamma(self, E):
        return self.a7/(self.a8 + E) + self.a9

    def betta(self, E):
        return self.a10/(self.a11 + E) + self.a12 + self.a13*E

    def Dif(self, C, temperature, ni):
        return self.b1 * exp(-self.b2 / (k * temperature)) + self.b3 * C / ni * exp(- self.b4 / (k * temperature)) + self.b5 * (C/ni)**2 * exp(- self.b6 / (k * temperature))


def gauss(x, Rp, dRp, Q):
    return Q/((2*pi)**0.5*dRp*1e-4)*exp(-(x - Rp)**2/(2*dRp**2))

def c_g(sc, Rp, dRp, Q, x, C, T, dt, ni):
    for i in range(0, len(x), 1):
            C[i] = Q/sqrt(2*pi*(dRp**2*1e-8+2*sc.Dif(C[i], T, ni)*dt))*exp(-(x[i] - Rp)**2/(2*dRp**2+4*sc.Dif(C[i], T, ni)*1e8*dt))
    return C

def c_time(sc, y, n, dx, T, ni):
    a = np.empty(n)
    b = np.empty(n)
    d = np.empty(n)
    r = np.empty(n)
    delta = np.empty(n)
    lyamda = np.empty(n)
    d[0] = 1
    a[0] = -1
    b[0] = 0
    r[0] = 0
    d[-1] = 0
    a[-1] = 1
    b[-1] = 0
    r[-1] = 0
    delta[0] = -d[0]/a[0]
    lyamda[0] = r[0]/a[0]
    dt = 1
    for i in range(1, n-1, 1):
        a[i] = -(2 + (dx ** 2 * 1e-8) / (sc.Dif(y[i], T, ni) * dt))
        r[i] = -(((dx ** 2 * 1e-8) * y[i]) / (sc.Dif(y[i], T, ni) * dt))
        b[i] = 1
        d[i] = 1
    for i in range(1, n, 1):
        delta[i] = -d[i] / (a[i] + b[i] * delta[i - 1])
        lyamda[i] = (r[i] - b[i] * lyamda[i - 1]) / (a[i] + b[i] * delta[i - 1])
    y[-1] = lyamda[-1]
    for i in range(n - 2, -1, -1):
        y[i] = delta[i] * y[i + 1] + lyamda[i]
    return y

def integrand(s, x, C, T, t, ni, r, R, sc):
    return exp(-(abs(s)-R)**2/r)*(exp(-(x-s)**2*1e-8/(4*sc.Dif(C, T, ni) * t))+exp(-(x+s)**2*1e-8/(4*sc.Dif(C, T, ni) * t)))

def c_special(sc, x, C, T, t, ni, dRp, Rp, dRp1, dRp2, Rm, Q, zagonka_method):
    if zagonka_method == "gauss":
        for i in range(0, len(x), 1):
            C[i] = Q / (2 * pi * sqrt(sc.Dif(C[i], T, ni) * t * 2*dRp**2)) * \
                   quad(integrand, 0, inf, args=(x[i], C[i], T, t, ni, 2*dRp**2, Rp, sc))[0]
    else:
        for i in range(0, len(x), 1):
            if x[i] <= Rm:
                r = 2*dRp1**2
            else:
                r = 2*dRp2**2
            C[i] = Q / (2 * pi * sqrt(sc.Dif(C[i], T, ni) * t * (dRp1+dRp2)**2)) * \
                   quad(integrand, 0, inf, args=(x[i], C[i], T, t, ni, r, Rm, sc))[0]
    return C

def N1(x, Rm, dRp2, dRp1, Q):
    return 2*Q/((2*pi)**0.5*(dRp1 + dRp2)*1e-4)*exp(-(x - Rm)**2/(2*dRp1**2))

def N2(x, Rm, dRp2, dRp1, Q):
    return 2*Q/((2*pi)**0.5*(dRp1 + dRp2)*1e-4)*exp(-(x - Rm)**2/(2*dRp2**2))

def pn_finder(C, Cp, x, n):
        xpn = 0
        for i in range(0, n):
            if C[i] >= Cp:
                xpn = x[i]
            else:
                break
        return xpn

class Sb(Si):

    def __init__(self):
        self.a1 = 0.000668
        self.a2 = 0.921
        self.a3 = 0.005072
        self.a4 = 0.000241
        self.a5 = 0.884
        self.a6 = 0.000923
        self.a7 = 195.1
        self.a8 = 339.7
        self.a9 = -0.091
        self.a10 = 47.33
        self.a11 = 81.17
        self.a12 = 2.692
        self.a13 = 0
        self.b1 = 0.214
        self.b2 = 3.65
        self.b3 = 15
        self.b4 = 4.08
        self.b5 = 0
        self.b6 = 0
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450
class As(Si):

    def __init__(self):
        self.a1 = 0.000688
        self.a2 = 0.983
        self.a3 = 0.003962
        self.a4 = 0.000402
        self.a5 = 0.874
        self.a6 = 0.000582
        self.a7 = 339.8
        self.a8 = 342
        self.a9 = -0.5051
        self.a10 = 38.73
        self.a11 = 61.70
        self.a12 = 2.559
        self.a13 = 0
        self.b1 = 0.66
        self.b2 = 3.44
        self.b3 = 12
        self.b4 = 4.05
        self.b5 = 0
        self.b6 = 0
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450
class P(Si):

    def __init__(self):
        self.a1 = 0.001555
        self.a2 = 0.958
        self.a3 = 0.000828
        self.a4 = 0.002242
        self.a5 = 0.659
        self.a6 = -0.003435
        self.a7 = 336.2
        self.a8 = 199.3
        self.a9 = -1.386
        self.a10 = 54.45
        self.a11 = 55.74
        self.a12 = 1.865
        self.a13 = 0.00482
        self.b1 = 3.85
        self.b2 = 3.66
        self.b3 = 4.44
        self.b4 = 4
        self.b5 = 44.2
        self.b6 = 4.37
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450
class B(Si):

    def __init__(self):
        self.a1 = 0.00969
        self.a2 = 0.767
        self.a3 = -0.01815
        self.a4 = 0.0521
        self.a5 = 0.216
        self.a6 = -0.0684
        self.a7 = 312.7
        self.a8 = 122.2
        self.a9 = -2.404
        self.a10 = 0
        self.a11 = 1
        self.a12 = 2.212
        self.a13 = 0.0195
        self.b1 = 0.037
        self.b2 = 3.46
        self.b3 = 0.72
        self.b4 = 3.46
        self.b5 = 0
        self.b6 = 0
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450

Sb = Sb()
As = As()
P = P()
B = B()

class Graph:
    Y = []
    X = []

    def draw(self, sc, x, C, dx, xmax, T, Rp, dRp, Q, Rm, dRp1, dRp2, t, ni, zagonka_method, razgonka_method, step, total):
        for i in range(len(x)):
            self.X.append(x[i])
            self.Y.append(C[i])
        
        printProgressBar(0, total, prefix='Progress:', suffix='Complete', length=50)
        for j in range(1, total+1, step):
            if razgonka_method == "analytic":
                if zagonka_method == "gauss":
                    C = c_g(sc, Rp, dRp, Q, x, C, T, j, ni)
            if razgonka_method == "numerical":
                C = c_time(sc, C, len(x), dx, T, ni)
            if razgonka_method == "special":
                C = c_special(sc, x, C, T, j, ni, dRp, Rp, dRp1, dRp2, Rm, Q, zagonka_method)
            time.sleep(0.1)
            printProgressBar(j, total, prefix='Progress:', suffix='Complete', length=50)
        print()
        plt.plot(self.X, self.Y, c='blue', label='C(x,0)')
        plt.plot(x, C, c='red', label='C(x,t)')
        plt.xlabel('x, мкм')
        plt.ylabel('Концентрация С')
        plt.legend()
        plt.xlim(0, xmax)
        plt.show()
        return C

Graph = Graph()

def implant(impurity, Q, E, T, t, Cp, n, xmax, razgonka_method, step, zagonka_method = 'gauss'):
    if impurity == 'Sb':
        sc = Sb
    if impurity == 'As':
        sc = As
    if impurity == 'B':
        sc = B
    if impurity == 'P':
        sc = P
    Rp = sc.Rp(E)
    dRp = sc.dRp(E)
    g = sc.gamma(E)
    b = sc.betta(E)
    ni = sc.ni(T)

    def solve(p):
        dRp1, dRp2 = p
        return (dRp ** (-3) * (dRp2 - dRp1) * (0.218 * dRp1 ** 2 + 0.362 * dRp1 * dRp2 + 0.218 * dRp2 ** 2) - g,
                -0.64 * (dRp2 - dRp1) ** 2 + (dRp1 ** 2 - dRp1 * dRp2 + dRp2 ** 2) - dRp ** 2)

    (dRp1, dRp2) = fsolve(solve, (1, 1))
    Rm = Rp - 0.8 * (dRp2 - dRp1)
    
    print("_________________________________________________________________________________________________")
    print("Rp = %.4f " % Rp, "dRp = %.4f " % dRp, "gamma = %.4f " % g, "betta = %.4f " % b)
    print("dRp2 = %.4f " % dRp2, "dRp1 = %.4f " % dRp1, "Rm = %.4f " % Rm)
    print("ni = " + str(sc.ni(T)))
    print("_________________________________________________________________________________________________")
    
    dx = xmax / n
    C = np.empty(n)
    x = np.empty(n)

    for i in range(0, n):
        if i == 0:
            x[i] = 0
        else:
            x[i] = x[i-1] + dx
        if zagonka_method == "gauss":
            C[i] = gauss(x[i], Rp, dRp, Q)
            #print(gauss(x[i], Rp, dRp, Q))
        if zagonka_method == "half-gauss":
            if x[i] <= Rm:
                C[i] = N1(x[i], Rm, dRp2, dRp1, Q)
            else:
                C[i] = N2(x[i], Rm, dRp2, dRp1, Q)

    if zagonka_method == 'half-gauss':
        qn1 = quad(N1, 0, Rm, args=(Rm, dRp2, dRp1, Q))[0]
        qn2 = quad(N2, Rm, xmax, args=(Rm, dRp2, dRp1, Q))[0]
        print('Q1 = ' + str(qn1 + qn2))

    if zagonka_method == 'gauss':
        q1 = quad(gauss, 0, xmax, args=(Rp, dRp, Q))[0]
        print('Q1 = ' + str(q1))

    if step >= 10:
        total = int(t*60 + 1)
    else:
        total = int(t*60)
    
    C = Graph.draw(sc, x, C, dx, xmax, T, Rp, dRp, Q, Rm, dRp1, dRp2, t, ni, zagonka_method, razgonka_method, step, total)
    q2 = simps(C, x)
    print("_________________________________________________________________________________________________")
    print('Q2 = ' + str(q2))

    Xpn = pn_finder(C, Cp, x, n)
    if Xpn > 0:
        print("_________________________________________________________________________________________________")
        print('Xpn = %.3f' % Xpn)
    else:
        print("_________________________________________________________________________________________________")
        print('Something Wrong')


# Задание параметров: примесь, доза имплантации, энергия, температура, время, концентрация примеси в подложке
#implant('Sb', 5e13, 150, 1273, 35, 1e17, 1000, 0.3, 'analytic', 1)
print('''
    ###Введение параметров###
    ''')
implant(
    input('Enter impurity (Sb, As, B, P) - '),
    float(input('Enter dose (1e12-1e14), cm^-2 - ')),
    float(input('Enter energy (100-200), keV - ')),
    float(input('Enter temperature (1000-1400), K - ')),
    float(input('Enter time (10-40), min - ')),
    float(input('Enter concentration (1e15-1e18), cm^-3 - ')),
    200,
	float(input('Enter max x value (0.2-1.0), mkm - ')),
	input("Enter type of dispersion (numerical, analytic, special) - "),
	1
)
input('Press ENTER to exit')
