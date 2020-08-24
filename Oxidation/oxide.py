print('''
    ----------------------------------------------
    Моделирование процесса окисления кремния (111)
    в сухом кислороде с параметрами окисления, вво
    димыми в консоли 
    ---------------------------- made by ineverbee
    ''')
print('''
    ###Установка зависимостей###
    ''')
import os
os.system('pip install -r requirements.txt')
import matplotlib.pyplot as plt
import math, time, sys
import numpy as np
from numpy import sqrt, arctan, inf, real, log, exp, pi

# Константы
q = 1.6e-19
k = 1.38e-23 / q  # эВ/К


# Расчёт параметров кремния
class Si:

    def __init__(self):
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450

    def Eg(self, temperature):
        return 1.17 - 4.73e-4 / (temperature + 636) * temperature ** 2

    def Ei(self, temperature):
        return self.Eg(temperature)/2 - k*temperature/4

    def Cminus(self, temperature):
        return exp((self.Ei(temperature)+0.57-self.Eg(temperature))/k/temperature)

    def Cplus(self, temperature):
        return exp((0.35-self.Ei(temperature))/k/temperature)

    def Cdoubleminus(self, temperature):
        return exp((2*self.Ei(temperature)+1.25-3*self.Eg(temperature))/k/temperature)

    def Vl(self, temperature):
        return 2620*exp(-1.1/k/temperature)

    def Vp(self, temperature):
        return 9.63e-16 * exp(2.83/k/temperature)

    def Vn(self, temperature, n):
        return (1 + self.Cplus(temperature)*self.ni(temperature)/n + self.Cminus(temperature)*n/self.ni(temperature) + self.Cdoubleminus(temperature)*(n/self.ni(temperature))**2)/(1+self.Cminus(temperature)+self.Cplus(temperature)+self.Cdoubleminus(temperature))

    def Nv(self, temperature):
        return 4.82e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def Nc(self, temperature):
        return 4.82e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * exp(
            -1 * self.Eg(temperature) / (2 * temperature * k))

def K(A, Ea, temperature):
    return A * exp(- Ea / k / temperature)

def X(t, A, B, xi):
    # постоянная времени
    # вместо xi при каждом запуске подставляется предыдущее значение x
    tau = (xi**2+A*xi)/B
    # толщина окисла x(t)
    x = (A/2)*(sqrt(1+(t+tau)/A**2*4*B)-1)
    return x

Si = Si()

def oxide(direction, T, P, time, C1, C2):
    sc = Si
    ni = sc.ni(T)
    
    # Параболическая константа
    B = K(12.9, 1.23, T)*(1 + sc.Vp(T)*ni**0.22)
    
    # "А" для первого слоя с концентрацией C1
    A1 = B/(K(1.04e5, 2, T)*(1 + sc.Vl(T)*(sc.Vn(T, C1)-1)))/P**0.75
    print('A1 = %0.3f' % A1)
    
    # "А" для следующего слоя с концентрацией C2
    A = B/(K(1.04e5, 2, T)*(1 + sc.Vl(T)*(sc.Vn(T, C2)-1)))/P**0.75
    print('A = %0.3f' % A)
    print('B = %0.3f' % B)

    xi0 = 0.035 # мкм  Начальный слой 20-50 нм, тк окисление происходит в сухом кислороде
    xi1 = 0.1/0.45 # мкм  Толщина первого слоя окисла
    n = time * 60 + 1
    dt = 1

    # Массивы времени и толщины
    t = []
    x = []
    x1 = []
    t1 = []

    # Присвоение начальных значений для первого слоя
    t1.append(0)
    x1.append(X(0, A1, B, xi0))

    # Цикл расчета толщины окисла для первого слоя
    while x1[-1] <= xi1:
        t1.append(t1[-1] + dt)
        x1.append(X(t1[-1], A1, B, xi0))

    t.append(t1[-1])
    x.append(X(t[-1]-t1[-1], A, B, x1[-1]))

    # Цикл расчета толщины окисла для следующего слоя
    while t[-1] <= n:
        t.append(t[-1] + dt)
        x.append(X(t[-1]-t1[-1], A, B, x1[-1]))

    plt.plot(t, x, c='black')
    plt.plot(t1, x1, c='yellow')
    plt.plot(t1[-1], x1[-1], c='purple', marker='o')
    plt.ylabel('Толщина слоя x, мкм')
    plt.xlabel('Время t, c')
    plt.xlim(0,t[-1])
    plt.ylim(0,x[-1])
    plt.show()

print('''
    ###Введение параметров###
    ''')
oxide(
    '111',
    float(input('Введите температуру окисления в Кельвинах (1200-1400)\n')),
    float(input('Введите давление, при котором проводится окисление, в атмосферах (1-2)\n')),
    float(input('Введите время окисления в минутах (10-40)\n')),
    float(input('Введите концентрацию первого слоя (1e16-1e18)\n')),
    float(input('Введите концентрацию последующего слоя (1e16-1e18)\n'))
)
input('Press ENTER to exit')