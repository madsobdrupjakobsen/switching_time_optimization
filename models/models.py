import numpy as np

class firstordermodel: 
    def __init__(self, pars):
        self.mu_IDLE = pars[0]
        self.mu_MELT = pars[1]
        self.lambda_IDLE = pars[2]
        self.lambda_MELT = pars[3]
        self.sigma = pars[4]
        self.R = pars[5]
        
        
    
        # Consider adding noise term
        
    def f_MELT(self, t,x):
        x = np.array(x)
        return(self.lambda_MELT * (self.mu_MELT - x))

    def f_IDLE(self, t,x):
        x = np.array(x)
        return(self.lambda_IDLE * (self.mu_IDLE - x))

    def f(self,t,x,tau_MELT,tau_IDLE):
        for i in range(tau_MELT.size-1):
            if tau_MELT[i] <= t and t < tau_IDLE[i+1]: # Check if MELT
                return(self.f_MELT(t,x))

            elif tau_IDLE[i] <= t and t < tau_MELT[i]: # Check if IDLE
                return(self.f_IDLE(t,x))

        # If last interval
        return(self.f_IDLE(t,x))
    
    def f_wrt_x(t,x,tau_MELT,tau_IDLE):
        for i in range(tau_MELT.size-1):
            if tau_MELT[i] <= t and t < tau_IDLE[i+1]: # Check if MELT
                return(-self.lambda_IDLE)

            elif tau_IDLE[i] <= t and t < tau_MELT[i]: # Check if IDLE
                return(-self.lambda_IDLE)

        # If last interval
        return(-self.lambda_IDLE)
    
    def h(self,x):
        y = x
        return y
    
    
    
    
class secondordermodel: 
    def __init__(self, pars):
        self.mu_IDLE = pars[0]
        self.mu_MELT = pars[1]
        self.omega_IDLE = pars[2]
        self.omega_MELT = pars[3]
        self.xi_IDLE = pars[4]
        self.xi_MELT = pars[5]
        self.sigma = np.exp(pars[6])
        self.R = np.exp(pars[7])

    def f_MELT(self,t,x):
        dxdt = np.zeros(2)
        dxdt[0] = -2 * self.xi_MELT * self.omega_MELT * (x[0]) - self.omega_MELT**2 * ((-x[1])-(-self.mu_MELT))
        dxdt[1] = -x[0]

        return dxdt

    def f_IDLE(self,t,x):
        dxdt = np.zeros(2)
        dxdt[0] = (-2 * self.xi_IDLE * self.omega_IDLE * (x[0]) - self.omega_IDLE**2 * ((-x[1])-(-self.mu_IDLE)))
        dxdt[1] = -x[0]

        return dxdt


    def f(self,t,x,tau_MELT,tau_IDLE):
        for i in range(tau_MELT.size-1):
            if tau_MELT[i] <= t and t < tau_IDLE[i+1]: # Check if MELT
                return(self.f_MELT(t,x))

            elif tau_IDLE[i] <= t and t < tau_MELT[i]: # Check if IDLE
                return(self.f_IDLE(t,x))

        # If last interval
        return(self.f_IDLE(t,x))
    
    def h(self,x):
        y = 1/1000 * x[1]
        return y