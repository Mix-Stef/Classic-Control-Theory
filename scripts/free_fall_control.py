import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#HYPERPARAMETERS
m = 1 #kg mass of falling object
R = 0.1 #m radius of falling object
b = 0.009546 # drag coefficient for sphere of radius 1 falling through air
h_sp = 20 #m set point of height
g = 9.81
h_0 = 30
err_0 = h_0 - h_sp
err_i0 = 0
u_0 = 0
A = np.pi * R * R

#CONTROLLER PARAMETERS
k_c = 5 
t_i = 200
k_p = 3
c_s = 0

#CONTROL UNIT PARAMETERS
k_s = 0.25 #control big rates of change of force
a_v = 10

def dYdt(Y, t):
    h, u, err_i = Y
    err = h - h_sp
    c = c_s + k_c * err + (1 / t_i) * err_i + k_p * u # PID
    F_control = k_s * c + a_v
    if F_control < 0:
        F_control = 0
    dhdt = u
    dudt = (m * g - F_control - b * u ** 2) / m
    deer_i_dt = err
    return [dhdt, dudt, deer_i_dt]

def main():
    t_span = np.linspace(0, 30, 100)
    sol = odeint(dYdt, (h_0, u_0, err_i0), t_span)
    h_span = sol[:, 0]
    u_span = sol[:, 1]
    err_i_span = sol[:, 2]
    err_span = h_span - h_sp
    c = c_s + k_c * err_span + (1 / t_i) * err_i_span + k_p * u_span # PID
    F_control = k_s * c + a_v
#    pressure = ((F_control / A) + 25) * 1e-5
    fig, ax = plt.subplots(1, 3, figsize=(25, 10))
    ax[0].plot(t_span, h_span)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("height")
    ax[0].grid()
    ax[1].plot(t_span, F_control)
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("controller force")
    ax[1].grid()
#    ax[2].plot(t_span, pressure)
#    ax[2].set_xlabel("time")
#    ax[2].set_ylabel("air pressure [bar]")
#    ax[2].grid()
    ax[2].plot(t_span, u_span)
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("velocity")
    ax[2].grid()
    plt.tight_layout()
    plt.savefig("free_fall.jpg")


if __name__ == "__main__":
    main()



