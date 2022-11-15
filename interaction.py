# Finite Volume code for solving three-wave interaction
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from utils import clear_datadir, save_data
from tqdm import tqdm

@ti.data_oriented
class ThreeWave:
    def __init__(self) -> None:
        # length unit: mum; time unit: ps
        
        # region
        # set the region large enough, so the bc will not affect the solution
        x_range = [0,2000] 
        dx = 1e-2
        nx = int(x_range[1]/dx)
        x = np.arange(dx/2, x_range[1], dx)

        self.nx = nx
        self.dx = dx
        self.x = ti.field(float, shape=x.shape)
        self.x.from_numpy(x)

        # parameters
        c = 3e2 # speed of light 
        vs = 0.6*c # soliton velocity
        wp_w0 = 0.1 # omega_p/omega_0
        a_p = 0.05 # initial pump amplitude
        a_s = 0.1 # initial seed amplitude
        x0 = 10 # pulse position
        ts = 0.01 # pulse time
        tr = 2.627*ts # rising time

        omega = ti.Vector([1.885e3, 1.6965e3]) # unit: 1/ps
        omega_p = omega[0]*wp_w0
        k = ti.sqrt(omega**2 - omega_p**2)/c
        k_p = k[0] + k[1] # resonance condition
        omega2 = omega[0] - omega[1] # resonance condition
        v2 = 0
        v = k*c**2/omega
        beta2 = c/4*k_p
        beta = beta2 * omega_p/omega

        # combine the v and beta params
        self.v = ti.Vector([v[0], v[1], v2])
        self.beta = ti.Vector([beta[0], beta[1], beta2])

        self.a_p = a_p
        self.k = k
        self.k_p = k_p
        self.omega = omega
        self.omega2 = omega2
        

        # solution a = [a0, a1, a2]^T
        # pump energy: a0**2
        # seed energy: a1**2
        # plasma energy: a2**2
        self.a = [ ti.Vector.field(n=2, dtype=float, shape=nx) for i in range(3) ]
        # save the intermediate values of multistep time integrator
        self.a_tmp = [ ti.Vector.field(n=2, dtype=float, shape=nx) for i in range(3) ]
        # to prevent data race
        self.a_new = [ ti.Vector.field(n=2, dtype=float, shape=nx) for i in range(3) ]

        # initial condition
        a = [
            a_p*np.heaviside((x-x0) - (v[0]+v[1])*tr, 0), # a0
            a_s*np.exp(-(x-x0)**2/(c*ts)**2), # a1
            1e-2*np.ones(nx) # a2
        ]
        for n in range(3):
            # need to split complex values to vec2
            an = np.column_stack([a[n].real, a[n].imag])
            self.a[n].from_numpy(an)

        # time
        self.t = ti.field(float,shape=())
        self.t[None] = 0.0
        self.dt = 0.9*dx/v[0]

    @ti.func
    def minmod(self, r:float) -> float:
        """ 
        1 argument version of minmod

        if r is nan or inf, the return is 1.
        """
        return ti.max(0, ti.min(1,r))

    @ti.func
    def F(self, u:ti.template(), n:int) -> ti.template():
        """ n-th component of flux term of the hyperbolic system """
        # return self.v[n]*u if n == 1 else -self.v[n]*u
        # 0-th index must be compile time constant integer
        # have to write it the following way
        v = 0.0
        if n == 0:
            v = -self.v[0]
        elif n == 1:
            v = self.v[1]
        elif n == 2:
            v = -self.v[2]
        return v*u
    
    @ti.func
    def F_hat(self, ul:ti.template(),ur:ti.template(), n:int) -> ti.template():
        """
        n-th component of numerical flux at an interface (Lax-Friedrichs flux)

        ul: left limit at the current interface
        ur: right limit at the current interface
        """
        F = ti.static(self.F)
        v = 0.0
        if n == 0:
            v = -self.v[0]
        elif n == 1:
            v = self.v[1]
        elif n == 2:
            v = -self.v[2]
        return 0.5*(F(ul,n)+F(ur,n)-ti.abs(v)*(ur-ul))

    @ti.func
    def rhs(self, a:ti.template(), t:float):
        a_new = ti.static(self.a_new)
        minmod, F_hat = ti.static(self.minmod, self.F_hat)
        dt, nx, dx, beta = self.dt, self.nx, self.dx, self.beta
        for i in range(2,nx-2): # Dirichlet boundary
            for n in ti.static(range(3)):
                # source term
                s = ti.math.vec2(0.0, 0.0)
                if n == 0:
                    s = beta[0]*a[1][i]*a[2][i]
                elif n == 1:
                    s = -beta[1]*a[0][i]*ti.math.cconj(a[2][i])
                elif n == 2:
                    s = -beta[2]*a[0][i]*ti.math.cconj(a[1][i])

                # linear reconstruction of cell values
                # and compute fluxes at cell interfaces
                u = a[n][i]
                u_prv, u_pprv = a[n][i-1], a[n][i-2]
                u_nxt, u_nnxt = a[n][i+1], a[n][i+2]
                
                r = (u_nxt-u)/(u-u_prv)
                r_prv = (u-u_prv)/(u_prv-u_pprv)
                r_nxt = (u_nnxt-u_nxt)/(u_nxt-u)

                # right interface
                ul = u + 0.5*(u-u_prv)*minmod(r)
                ur = u_nxt - 0.5*(u_nxt-u)*minmod(r_nxt)
                FR = F_hat(ul, ur, n)

                # left interface
                ur = u - 0.5*(u-u_prv)*minmod(r)
                ul = u_prv + 0.5*(u_prv-u_pprv)*minmod(r_prv)
                FL = F_hat(ul, ur, n)

                a_new[n][i] = dt * (-(FR-FL)/dx + s)

    @ti.func
    def rk2(self, t: float):
        """ 2nd order Runge-Kutta time integration """
        a, a_tmp, a_new, rhs = ti.static(self.a, self.a_tmp, self.a_new, self.rhs)
        dt, nx = self.dt, self.nx

        rhs(a, t) # k1 = a_new
        for i in range(nx):
            for n in ti.static(range(3)):
                # a+k1/2
                a_tmp[n][i] = a_new[n][i]/2 + a[n][i]
        
        rhs(a_tmp, t+dt/2) # a_tmp = k2
        for i in range(nx):
            for n in ti.static(range(3)):
                a[n][i] += a_new[n][i]

    @ti.kernel
    def advance(self):
        t = ti.static(self.t)
        dt = self.dt

        self.rk2(t[None])
        t[None] += dt # update current time

    def run(self):
        tf = 1
        t_range = np.linspace(0,tf,int(tf/self.dt)) 

        print(f"total {t_range.size} frames \n")

        clear_datadir()
        save_data(0, self.a, to_numpy=True)
        for n, t in enumerate(tqdm(t_range[1:])):
            self.advance()

            if (n % 100 == 0):
                save_data(n, self.a, to_numpy=True)
        np.save("data/x", self.x.to_numpy())
        np.save("data/t", t_range)


if __name__ == '__main__':
    ti.init(ti.gpu, default_fp=ti.f64)
    sim = ThreeWave()
    sim.run()