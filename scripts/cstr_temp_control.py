import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#HYPERPARAMETERS
k_0 = 0.00017
V = 5.119
E = 11273
R = 1.987
DH = 590000
cp_a = 40
cp_b = 8.38
cp_w = 18
k_c = -0.5
t_i = 60
c_s = 55
k_v = -25
UA = 35.85
T_c0 = 298
n_a0 = 9.044
n_b0 = 33
n_w0 = 103.7
x_0 = 0
T_0 = 448
err_i0 = 0
err_0 = 0 
T_sp = 448
a_v = 598
c_a0 = n_a0 / V
dcp = n_a0 * cp_a + n_b0 * cp_b + n_w0 * cp_w



def dYdt(Y, t):
    x, T, err_i = Y
    err = T_sp - T
    c = c_s + k_c * err + (k_c / t_i) * err_i
    t_c = k_v * c + a_v
    if t_c < 298:
        t_c = 298
    if t > 45 and t < 55:
        qr = 0
    else:
        qr = UA * (T - t_c)
    ra = k_0 * np.exp((E / R) * ((1 / 461) - (1/ T))) * (c_a0 ** 2) * (1 - x) * ((n_b0 / n_a0) - 2 * x)
    dTdt = (ra * V * (DH) - qr) / dcp
    derr_i_dt = err
    return [ra * V / n_a0, dTdt, derr_i_dt]


def main():
    t_span = np.linspace(0, 8000, 1500)
    sol = odeint(dYdt, (x_0, T_0, err_i0), t_span)
    x_conc = sol[:, 0]
    temp = sol[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(t_span, x_conc, c='red')
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("Conversion")
    ax[0].grid()
    ax[1].plot(t_span, temp - 273, c='blue')
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Temperature")
    ax[1].grid()
    plt.tight_layout()
    plt.savefig("cstr_control.jpg")

if __name__ == "__main__":
    main()
