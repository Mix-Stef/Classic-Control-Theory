import sympy as sp
import numpy as np
from sympy import pprint, Poly, roots, simplify, solve, N, re, im, fraction, inverse_laplace_transform, lambdify, limit, cos
from typing import List
from sympy.matrices import Matrix
import numbers
import matplotlib.pyplot as plt


TIME_DOMAIN = np.linspace(0, 6, 2000)


class SISO_TransferFunction():
    def __init__(self, numerator: List[float], denominator: List[float]) -> None:
        self.numerator = numerator
        self.denominator = denominator
        #stupid fix because algebra is not correct if self.denominator[-1] == 0
        if self.denominator[-1] == 0:
            self.denominator[-1] = np.nextafter(0, 1)
        self.s = sp.Symbol("s") #Frequency variable
        self.t = sp.Symbol("t") #Time variable
        self.num_poly = Poly(self.numerator, self.s)
        self.den_poly = Poly(self.denominator, self.s)
        self.tf = self.num_poly / self.den_poly
        #In case the coefficients involve parameters
        self.param_types = [sp.core.symbol.Symbol, sp.core.add.Add, sp.core.mul.Mul] 
        self.num_contains_parameter = np.any([type(x) in self.param_types for x in self.numerator])
        self.den_contains_parameter = np.any([type(x) in self.param_types for x in self.denominator])
    
    def sub(self, params: list):
        #Params: list of tuples (symbol, value), as per sympy docs
        self.tf.subs(params)

    #Display the transfer function symbolically
    def show(self) -> None:
        #Over engineering to compensate for infinitesimal that may arise from the stupid fix
        dummy_num_poly, dummy_den_poly = Poly(list(map(lambda x: round(x, 2) if not self.num_contains_parameter else x * 1 ,self.numerator)) ,self.s), Poly(list(map(lambda x: round(x, 2) if not self.den_contains_parameter else x * 1 ,self.denominator)) ,self.s)
        dummy_tf = dummy_num_poly / dummy_den_poly 
        pprint(dummy_tf, use_unicode=False)

    #Functions for finding roots and poles of transfer function
    def tf_roots(self) -> List[float]:
        return list(map(lambda x: round(N(x), 4), solve(self.num_poly)))

    def tf_poles(self) -> List[float]:
        return list(map(lambda x: round(N(x), 4), solve(self.den_poly)))

        
    #Check stability of transfer function
    def is_stable(self) -> bool:
        #MUST FIX ACCORDING TO THEORY
        poles = self.tf_poles()
        real_parts = list(map(re, poles))
        if np.all([re < 0 for re in real_parts]):
            return True
        return False

    #Functions for transfer function algebra
    def get_new_tf(self, g_):
        num, den = fraction(g_)
        num_coeffs, den_coeffs = Poly(num,self.s).coeffs(), Poly(den, self.s).coeffs()
        new_g = SISO_TransferFunction(num_coeffs, den_coeffs)
        return new_g

    def multiply_with(self, tf_):
        if isinstance(tf_, numbers.Number):
            g_ = (self.tf * tf_).simplify()
            return self.get_new_tf(g_)
        else:
            g_ = (self.tf * tf_.tf).simplify()
            return self.get_new_tf(g_)

    def add(self, tf_):
        if isinstance(tf_, numbers.Number):
            g_ = (self.tf + tf_).together()
            return self.get_new_tf(g_)
        else:
            g_ = (self.tf + tf_.tf).together()
            return self.get_new_tf(g_)

    def subtract(self, tf_):
        if isinstance(tf_, numbers.Number):
            g_ = (self.tf - tf_).together()
            return self.get_new_tf(g_)
        else:
            g_ = (self.tf - tf_.tf).together()
            return self.get_new_tf(g_)

    def divide_by(self, tf_):
        if isinstance(tf_, numbers.Number):
            g_ = (self.tf / tf_).simplify()
            return self.get_new_tf(g_)
        else:
            g_ = (self.tf / tf_.tf).simplify()
            return self.get_new_tf(g_)

    #Function to go from frequency domain (s) to time domain (t) via inverse Laplace transform 
    def back_to_time(self):
        return inverse_laplace_transform(self.tf, self.s, self.t).simplify()

    #Get the values of inverse Laplace transform at time-domain 
    def get_time_values(self) -> List[float]:
        f_t = lambdify(self.t, self.f, modules=['numpy'])
        return f_t(TIME_DOMAIN)
    
    #Check for convergence and find limit value
    def limit_value(self) -> float:
        lim = limit(self.f, self.t, np.infty)
        return lim

    def converges(self) -> tuple([bool, float]):
        lim = self.limit_value()
        if isinstance(lim, numbers.Number):
            return True, lim
        return False, None

    #Functions to compute rise-time, peak-time, settling-time and overshoot
    def rise_time(self) -> float:
        f_values = self.get_time_values()
        convergence, lim = self.converges()
        if convergence:
            ten_percent, ninety_percent = 0.1 * lim, 0.9 * lim
            #Search for 10% time
            i = 0
            tol = 1e-2
            while abs(f_values[i] - ten_percent) > tol:
                i += 1
            ten_percent_time = TIME_DOMAIN[i]
            #Search for 90% time
            while abs(f_values[i] - ninety_percent) > tol:
                i += 1
            ninety_percent_time = TIME_DOMAIN[i]
            return ninety_percent_time - ten_percent_time
        print("Function does not converge")
        return

    def peak_time(self) -> float:
        f_values = self.get_time_values()
        max_value_index = np.argmax(f_values)
        return TIME_DOMAIN[max_value_index]

    def settling_time(self) -> float: # margin & tol may crash it
        #Increasing time steps of time domain may fix it
        f_values = self.get_time_values()
        convergence, lim = self.converges()
        if convergence:
            margin = 0.95 * lim
            i = 0
            tol = 1e-2
            while i < len(TIME_DOMAIN):
                i += 1
                if abs(f_values[i] - margin) < tol:
                    if np.all([abs(f_values[i:i+5] - margin) < tol]):
                        return TIME_DOMAIN[i]
        print("Function does not converge")
        return

    def overshoot(self) -> float:
        convergence, lim = self.converges()
        if convergence:
            f_t_vals = self.get_time_values()
            return round((np.max(f_t_vals) - lim) / lim, 5)

    #Plot the response at time-domain
    def plot_response(self, filename: str) -> None:
        f_t = lambdify(self.t, self.back_to_time(), modules=['numpy'])
        f_t_vals = f_t(TIME_DOMAIN)
        plt.plot(TIME_DOMAIN, f_t_vals)
        plt.xlabel("time")
        plt.ylabel("response")
        plt.grid()
        plt.savefig(filename + ".jpg")

class MIMO_TransferFunction():
    def __init__(self, A: Matrix, B: Matrix, C: Matrix, D: Matrix):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.s = sp.Symbol("s")
        self.N = len(self.A)
        self.transfer_function = np.matmul(np.matmul(self.C, (self.s * sp.eye(self.N) - self.A).inv()), self.B) + self.D
        
    def show_tf(self) -> None:
        pprint(list(map(simplify,self.transfer_function)), use_unicode=False)

g1 = SISO_TransferFunction(numerator=[2], denominator=[1, 4, 0])
g1.show()
