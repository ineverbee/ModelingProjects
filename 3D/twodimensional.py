print('''
    ----------------------------------------
    Моделирование процесса диффузии сурьмы в
    кремнии после имплантации с параметрами, 
    вводимыми в консоли (3D version)
    ---------------------- made by ineverbee
    ''')
print('''
    ###Установка зависимостей###
    ''')
import os
os.system('pip install -r requirements.txt')
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erfc
from matplotlib import cm

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

# Расчёт параметров кремния, проецированного пробега ионов,
# разброса проецированного пробега ионов и ассиметрии распределения
class Si:

    def Eg(self, temperature):
        return 1.17 - 4.73e-4 / (temperature + 636) * temperature ** 2

    def Nv(self, temperature):
        return 4.82e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def Nc(self, temperature):
        return 4.82e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * math.exp(
            -1 * self.Eg(temperature) / (2 * temperature * k))

    def Rp(self, E):
        return self.a1*E**self.a2 + self.a3

    def dRp(self, E):
        return self.a4*E**self.a5 + self.a6

    def gamma(self, E):
        return self.a7/(self.a8 + E) + self.a9

    def betta(self, E):
        return self.a10/(self.a11 + E) + self.a12 + self.a13*E

    def dRpl(self, E):
        return np.sqrt(2*self.Rp(E)/(self.Se+self.Sn)*E - self.dRp(E)**2 - self.dRp(E)**2)


# Коэффициент диффузии сурьмы в кремнии
def Dif(C, temperature, ni):
    return 0.214 * math.exp(-3.65/(k*temperature)) + 15 * C / ni * math.exp(-4.08/(k*temperature))

def c_time(y, n_stroki, n_stolbca, dx, dy, T, ni):

    for o in range(1, n_stolbca-1):
        a = np.empty(n_stroki)
        b = np.empty(n_stroki)
        d = np.empty(n_stroki)
        r = np.empty(n_stroki)
        delta = np.empty(n_stroki)
        lyamda = np.empty(n_stroki)
        n = n_stroki
        if n == 40:
            d[0] = 1
            a[0] = -1
        else:
            d[0] = 0
            a[0] = 1
        b[0] = 0
        r[0] = 0
        d[-1] = 0
        a[-1] = 1
        b[-1] = 0
        r[-1] = 0
        delta[0] = -d[0]/a[0]
        lyamda[0] = r[0]/a[0]
        dt = 1
        for i in range(1, n_stroki-1, 1):
            a[i] = -(2 + (dx ** 2 * 1e-8) / (Dif(y[o][i], T, ni) * dt))
            r[i] = -dx**2/dy**2*(y[o+1][i]-(2-(dy ** 2 * 1e-8)/(Dif(y[o][i], T, ni) * dt))*y[o][i]+y[o-1][i])
            b[i] = 1
            d[i] = 1
        for i in range(1, n_stroki, 1):
            delta[i] = -d[i] / (a[i] + b[i] * delta[i - 1])
            lyamda[i] = (r[i] - b[i] * lyamda[i - 1]) / (a[i] + b[i] * delta[i - 1])
        y[o][-1] = lyamda[-1]
        for i in range(n_stroki - 2, -1, -1):
            y[o][i] = delta[i] * y[o][i + 1] + lyamda[i]

    for p in range(1, n_stroki-1):
        a = np.empty(n_stolbca)
        b = np.empty(n_stolbca)
        d = np.empty(n_stolbca)
        r = np.empty(n_stolbca)
        delta = np.empty(n_stolbca)
        lyamda = np.empty(n_stolbca)
        n = n_stolbca
        if n == 40:
            d[0] = 1
            a[0] = -1
        else:
            d[0] = 0
            a[0] = 1
        b[0] = 0
        r[0] = 0
        d[-1] = 0
        a[-1] = 1
        b[-1] = 0
        r[-1] = 0
        delta[0] = -d[0]/a[0]
        lyamda[0] = r[0]/a[0]
        dt = 1
        for i in range(1, n_stolbca-1, 1):
            a[i] = -(2 + (dx ** 2 * 1e-8) / (Dif(y[i][p], T, ni) * dt))
            r[i] = -dx**2/dy**2*(y[i][p+1]-(2-(dy ** 2 * 1e-8)/(Dif(y[i][p], T, ni) * dt))*y[i][p]+y[i][p-1])
            b[i] = 1
            d[i] = 1
        for i in range(1, n_stolbca, 1):
            delta[i] = -d[i] / (a[i] + b[i] * delta[i - 1])
            lyamda[i] = (r[i] - b[i] * lyamda[i - 1]) / (a[i] + b[i] * delta[i - 1])
        y[-1][p] = lyamda[-1]
        for i in range(n_stolbca - 2, -1, -1):
            y[i][p] = delta[i] * y[i + 1][p] + lyamda[i]
    return y

def two_dimensional(x, y, Q, dRp, Rp, dRpl, a):
    return Q/np.sqrt(2*np.pi)/dRp/1e-4*np.exp(-(x-Rp)**2/2/dRp**2)/2*(erfc((y-a)/np.sqrt(2)/dRpl)-erfc((y+a)/np.sqrt(2)/dRpl))

# Параметры примеси и подложки
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
        # for dRpl
        self.Se = 520 # кэВ/мкм для 150 кэВ
        self.Sn = 1833.33 # кэВ/мкм для 150 кэВ
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450

Sb = Sb()

class Graph:
    Y = []
    X = []

    def draw(self, start, end, x, dx, xmax, T, Rp, dRp, dRpl, Q, ni, t):
        for o in range(-len(x)+1, len(x)-1, 1):
            if o >= 0:
                self.Y.append(2*x[o])
            else:
                self.Y.append(-2*x[-o])
        
        for o in range(0, len(x), 1):
            self.X.append(x[o])
        dy = self.Y[30]-self.Y[29]
        X, Y = np.meshgrid(self.X, self.Y)
        Z = X**2 - Y**2
        for i in range(0, len(self.Y), 1):
            for j in range(0, len(self.X), 1):
                Z[i][j] = two_dimensional(X[i][j], Y[i][j], Q, dRp, Rp, dRpl, xmax/2)

        fig = plt.figure('graph')
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.view_init(azim=-115, elev=25)
        ax1.set_xlim(0, xmax)
        ax1.set_ylim(-xmax*2, xmax*2)
        ax1.set_zlim(0, 7e18)
        ax1.set_xlabel('x, мкм')
        ax1.set_ylabel('y, мкм')
        ax1.set_zlabel('Концентрация, С')
        surf = ax1.plot_surface(X, Y, Z, cmap='autumn', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        timing = int(t*60)
        printProgressBar(0, timing - 1, prefix='Progress:', suffix='Complete', length=50)
        for j in range(1, timing, 1):
            Z = c_time(Z, len(Z[1]),len(Z[:,1]), dx, dy, T, ni)
            printProgressBar(j, timing - 1, prefix='Progress:', suffix='Complete', length=50)
        print()

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.view_init(azim=-115, elev=25)
        ax2.set_xlim(0, xmax)
        ax2.set_ylim(-xmax * 2, xmax * 2)
        ax2.set_zlim(0, 7e18)
        ax2.set_xlabel('x, мкм')
        ax2.set_ylabel('y, мкм')
        ax2.set_zlabel('Концентрация, С')
        surf = ax2.plot_surface(X, Y, Z, cmap='autumn', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


Graph = Graph()


def implant(impurity, Q, E, T, t):
    if impurity == 'Sb':
        sc = Sb

    Rp = sc.Rp(E)
    dRp = sc.dRp(E)
    dRpl = sc.dRpl(E)
    g = sc.gamma(E)
    b = sc.betta(E)
    ni = sc.ni(T)
    print("_________________________________________________________________________________________________")
    print("Rp = %.4f " % Rp, "dRp = %.4f " % dRp, "dRpl = %.4f " % dRpl, "gamma = %.4f " % g, "betta = %.4f " % b)
    print("ni = " + str(sc.ni(T)))
    print("_________________________________________________________________________________________________")
    n = 40
    xmax = 0.25
    dx = xmax / n
    x = np.empty(n)
    for i in range(0, n):
        if i == 0:
            x[i] = 0
        else:
            x[i] = x[i-1] + dx
    Graph.draw(0, n, x, dx, xmax, T, Rp, dRp, dRpl, Q, ni, t)


# Задание параметров: примесь, доза имплантации, энергия, температура, время
print('''
    ###Введение параметров###
    ''')
implant(
    'Sb',
    float(input('Enter dose (1e12-1e14), cm^-2 - ')),
    float(input('Enter energy (100-200), keV - ')),
    float(input('Enter temperature (1000-1400), K - ')),
    float(input('Enter time (10-40), min - ')),
)
input('Press ENTER to exit')