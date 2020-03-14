import numpy as np
from scipy.integrate import solve_ivp

def stochasticSimulation(model,switches,x0,tf,dt):
    
    # Compute regimes
    tau_MELT_all, tau_IDLE_all = derive_regimes(switches,tf,1)
    
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


def simulate_MPC(model,x0,n_days,nswitches,prices,k,k_MELT,k_IDLE,dt,start_date,seed,save_to_file):   
    np.random.seed(seed)

    # Simulate parameters
    #n_days = 30
    #dt = 0.01
    #start_date = '01-08-2013'

    # Initialize process
    #x0 = np.array([2.6659426, 899.8884004])
    nx = x0.shape[0]
    
    tf_ph = 48 * 60
    tf_sim = 24 * 60

    # Initialzie history
    history = {}
    history['X'] = np.zeros([n_days, nx, 60 * 24])
    history['Z'] = np.zeros([n_days, 60 * 24])
    history['Z_ode'] = np.zeros([n_days, 60 * 24])
    history['T'] = np.zeros([n_days, 60 * 24])

    history['SWITCHES'] = np.zeros([n_days, nswitches])
    history['PRICES'] = np.zeros([n_days, 48])

    history['expected_price'] = np.zeros(n_days)
    history['true_price'] = np.zeros(n_days)


    idx_offset = np.where(prices.index == start_date)[0][0]
    for day in range(n_days):
        print('Simulating day ' + str(day))

        # Extract variables related to the day
        idx = np.arange(48) + day * 48 + idx_offset
        future_days = np.array(prices.index[idx])
        future_hours = np.array(prices['Hours'][idx])
        future_price = np.array(prices['DK1'][idx]).astype(np.float)


        # ---------------------------
        # Compute optimal switches over 2 days
        # ---------------------------

        # To be changed to C++ optimizer
        switch_opt = np.sort(np.random.uniform(0,tf_ph,nswitches))


        # ---------------------------
        # Simulate 1 day
        # ---------------------------
        switch_sim = switch_opt
        #nswicthes_switches_in_ph = np.sum(switch_opt < tf_sim) + np.sum(switch_opt < tf_sim) % 2
        #switch_sim = switch_opt[switch_opt < tf_sim]

        # Consider only saving every 1 minutes of the simulation
        T_tmp, X_tmp, Y_tmp, Z_tmp = stochasticSimulation(model,switch_sim,x0,tf_sim,dt)
        T_tmp_ode, X_tmp_ode, Z_tmp_ode = solve_ivp_discrete(model,x0,switch_sim,tf_sim,T_tmp)


        # ---------------------------
        # Compute Cost
        # ---------------------------
        dap = 1/1000000 * future_price[:24]
        cost_true, cost_acc_true = cost(Z_tmp,T_tmp,switch_sim,dap,k,k_MELT,k_IDLE)
        cost_exp, cost_acc_exp = cost(Z_tmp_ode,T_tmp_ode,switch_sim,dap,k,k_MELT,k_IDLE)



        # ---------------------------
        # Update variables
        # ---------------------------

        # Process
        x0 = np.array([X_tmp[i][-1] for i in range(nx)])

        # Update monitored variables
        history['expected_price'][day] = cost_true[-1]
        history['true_price'][day] = cost_exp[-1]

        history['Z'][day] = Z_tmp[::int(1/dt)]
        history['Z_ode'][day] = Z_tmp_ode[::int(1/dt)]
        history['X'][day,:] = X_tmp[:,::int(1/dt)]
        history['T'][day] = T_tmp[::int(1/dt)] # Same in each day
        history['SWITCHES'][day,:] = switch_opt
        history['PRICES'][day,:] = future_price

    filenname = '../results/sim_history/history_' + start_date + '_' + str(n_days) + '_days' + '_seed_' + str(seed) + '.npy'
    np.save(filenname,history)
    
    return history




def solve_ivp_discrete(model,x0,switches,tf,t_plot):
    nx = x0.size
    if nx > 1:
        start = [x0[i] for i in range(nx)]
    else:
        start = [x0]


    tau_MELT_all, tau_IDLE_all  = derive_regimes(switches,tf,1)
    switches_ext = np.append(np.insert(switches,0,0),tf)

    T = np.array([])
    X = np.zeros((nx,1))


    # Loop through all regimes
    for i in range(len(switches_ext)-1):


        if (t_plot is None): 

            t_eval = None
        else:

            t_eval = t_plot[ (switches_ext[i] <= t_plot) & (t_plot < switches_ext[i+1])]
            if (len(t_eval) != 0):

                sol = solve_ivp(model.f, 
                        [switches_ext[i], 
                         switches_ext[i+1]], 
                        start, 
                        args=(tau_MELT_all,tau_IDLE_all), 
                        t_eval = t_eval)

                T = np.concatenate((T,sol.t))
                X = np.hstack([X,sol.y])


                start = [sol.y[j][-1] for j in range(nx) ]


    X = np.array([X[i][1:] for i in range(nx)])
    Z = model.h(X)
    
    return np.squeeze(T), np.squeeze(X), np.squeeze(Z)
    
    return(t,y)

def sol_ivp_wrapper(model,x0,switches,tf,t_plot):
    tau_MELT_all, tau_IDLE_all  = derive_regimes(switches,tf,1)
    x0 = [x0[i] for i in range(len(x0))]
    
    sol = solve_ivp(model.f, [0, tf], x0, args=(tau_MELT_all,tau_IDLE_all),t_eval = t_plot)
    
    X = sol.y
    Z = model.h(X)
    T = sol.t
    return np.squeeze(T), np.squeeze(X), np.squeeze(Z)

def derive_regimes(switches,tf,endpoints):
    n_switches = switches.size
    n_cycles = int(n_switches/2)
    tau_MELT = switches[np.arange(0,n_switches,2)]
    tau_IDLE = switches[np.arange(1,n_switches,2)]
    
    if endpoints:
        tau_IDLE = np.insert(tau_IDLE,0,0)
        tau_MELT = np.append(tau_MELT,tf)
    
    return tau_MELT, tau_IDLE 

def mergeSwitchCont(switches):
    n_s = int(len(switches)/2)
    melt = switches[:n_s]
    idle = switches[n_s:]
    return(np.insert(idle, np.arange(len(melt)), melt))

# ENERGY COST
def smooth_dap(t,dap):
    dat = 60 * np.arange(dap.size + 1); dat[0] = dat[0] - 60; dat[-1] = dat[-1] + 60
    
    _dap = 0
    for k in range(24):
            _dap += dap[k] / ((1. + np.exp(np.minimum(-1 * (t - dat[k]), 15.0 ))) *
                             (1. + np.exp( np.minimum(1. * (t - dat[k + 1]), 15.))))
    return _dap


def smooth_regime(T,switches):
    n_s = int(len(switches)/2)
    tau_MELT, tau_IDLE  = switches[:n_s] , switches[n_s:] #derive_regimes(switches,T[-1],0)
    regime = 0
    for k in range(len(tau_MELT)):
            regime += 1/ ((1 + np.exp(np.minimum(-1. * (T - tau_MELT[k]), 15.0 ))) *
                             (1 + np.exp( np.minimum(1. * (T - tau_IDLE[k]), 15.))))
            
    return regime


def cost(traj,T,switches,dap,k,k_MELT,k_IDLE):
    
    
    dt = np.diff(T)
    regime = smooth_regime(T,switches)
    _dap = smooth_dap(T,dap)
    cost_momental = _dap * (k * traj + k_IDLE * (regime <= 0.5) + k_MELT * (regime > 0.5)) * 1/60 # 1/60 to Compensate for unit change
    cost_acc = np.cumsum(cost_momental[1:] * dt)

    #plt.plot(cost_momental)
    return cost_momental, cost_acc


# MATH
def sigmoid(x,slope,offset):
    return 1./(1. + np.exp(-(slope * (x - offset))))