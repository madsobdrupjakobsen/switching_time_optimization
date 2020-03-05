import numpy as np
from scipy.integrate import solve_ivp

def stochasticSimulation(model,switches,x0,tf,dt):
    
    # Compute regimes
    tau_IDLE_all, tau_MELT_all = derive_regimes(switches,tf,1)
    
    # Model functions
    f = model.f
    h = model.h
    sigma = model.sigma
    R = model.R
    
    # Initialize vairables
    x = x0
    N = int(tf/dt)
    nx = x.size
    nz = h(x).size
    X = np.zeros((nx,N))
    Y = np.zeros((nz,N))
    Z = np.zeros((nz,N))
    T = np.zeros((N))


    # Draw noise
    dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(nx), np.eye(nx), N)
    dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(nx), np.eye(nx), N)
    v = np.sqrt(R) * np.random.multivariate_normal(np.zeros(nz), np.eye(nz), N)


    # Run simulation
    t = 0
    for k in range(N):
        dx = f(t,x,tau_MELT_all,tau_IDLE_all)
        dw = sigma * dW[k]

        x = x + dx * dt + dw
        z = h(x)
        t += dt

        # Save current state
        X[:,k] = x
        Z[:,k] = z
        Y[:,k] = z + v[k]
        T[k] = t
        
    return np.squeeze(T), np.squeeze(X), np.squeeze(Y), np.squeeze(Z)



def derive_regimes(switches,T,endpoints):
    n_switches = switches.size
    n_cycles = int(n_switches/2)
    tau_MELT = switches[np.arange(0,n_switches,2)]
    tau_IDLE = switches[np.arange(1,n_switches,2)]
    
    if endpoints:
        tau_IDLE = np.insert(tau_IDLE,0,0)
        tau_MELT = np.append(tau_MELT,T)
    
    return tau_IDLE, tau_MELT

def solve_ivp_discrete(model,x0,switches,tf,t_plot):
    nx = x0.size
    start = [x0[i] for i in range(nx)]


    tau_IDLE_all, tau_MELT_all = derive_regimes(switches,tf,1)
    switches_ext = np.append(np.insert(switches,0,0),tf)

    T = np.array([])
    X = np.zeros((nx,1))


    # Loop through all regimes
    for i in range(len(switches_ext)-1):
        #print("i")
        #print(i)

        if (t_plot is None): 

            t_eval = None
        else:

            t_eval = t_plot[ (switches_ext[i] < t_plot) & (t_plot < switches_ext[i+1])]
            if (len(t_eval) == 0): t_eval = None



        sol = solve_ivp(model.f, 
                        [switches_ext[i], 
                         switches_ext[i+1]], 
                        start, 
                        args=(tau_MELT_all,tau_IDLE_all), 
                        t_eval = t_eval)

        T = np.concatenate((T,sol.t))
        X = np.hstack([X,sol.y])


        start = [sol.y[i][-1] for i in range(nx) ]


    X = np.array([X[i][1:] for i in range(nx)])
    Z = model.h(X)
    
    return np.squeeze(T), np.squeeze(X), np.squeeze(Z)
    
    return(t,y)

def sol_ivp_wrapper(model,x0,switches,tf,t_plot):
    tau_IDLE_all, tau_MELT_all = derive_regimes(switches,tf,1)
    x0 = [x0[i] for i in range(len(x0))]
    
    sol = solve_ivp(model.f, [0, tf], x0, args=(tau_MELT_all,tau_IDLE_all),t_eval = t_plot)
    
    X = sol.y
    Z = model.h(X)
    T = sol.t
    return np.squeeze(T), np.squeeze(X), np.squeeze(Z)


# ENERGY COST
def smooth_dap(dap,t):
    dat = 60 * np.array(range(dap.size + 1)); dat[0] = dat[0] - 60; dat[-1] = dat[-1] + 60
    
    _dap = 0
    for k in range(24):
            _dap += dap[k] / ((1. + np.exp(np.minimum(-1 * (t - dat[k]), 15.0 ))) *
                             (1. + np.exp( np.minimum(1. * (t - dat[k + 1]), 15.))))
    return _dap


def smooth_regime(T,switches):
    tau_IDLE, tau_MELT = derive_regimes(switches,T,0)
    regime = 0
    for k in range(len(tau_MELT)):
            regime += 1/ ((1 + np.exp(np.minimum(-1. * (T - tau_MELT[k]), 15.0 ))) *
                             (1 + np.exp( np.minimum(1. * (T - tau_IDLE[k]), 15.))))
            
    return regime