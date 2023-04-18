
import numpy as np
import pandas as pd

class GenData():
    # quadratic map
    length = 100
    c = 1.2
    x_init = 0.8

    # mackey glass
    beta = 0.2
    gamma = 0.1
    tau = 17
    n = 10

    # rossler
    # length = 100000
    a = 0.13
    b = 0.2
    c = 6.5
    init_time = 0
    end_time = 32*np.pi

    # lorenz
    sigma = 10
    rho = 28
    beta = 2.667
    init = [0, 1, 1.05]
    # init_time = 0
    # end_time = 100

    def quadratic_map(ii, x_init, length= 1, c= 1.2):
        x = c - x_init**2
        qx = [x]
        for i in range(length):
            x = c - x**2
            qx.append(x)
        return pd.DataFrame.from_dict({'t': np.linspace(0, length, length+1), 'x': qx})
    
    def mackey_glass(ii, x_init= None, length= 1, beta= 0.2, gamma= 0.1, tau= 17, n= 10):
        # x_init= [
        #     0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 
        #     1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759
        # ]
        assert length > tau
        # add 100 warm up points to length
        warmup = 100
        length += warmup
        mgx = []
        if x_init is not None:
            assert len(x_init) == tau+1
            mgx += x_init
        else:
            mgx += np.random.rand(tau+1,).tolist()
        for i in range(tau, length):
            new_x = mgx[i] + beta * (mgx[i-tau]/(1 + mgx[i-tau]**n)) - gamma * mgx[i]
            mgx.append(new_x)
        t = np.linspace(0, length, length+1, endpoint=False)
        return pd.DataFrame.from_dict({'t': t, 'x': np.array(mgx)})
    
    def rossler(ii, length= 1, a= 0.13, b= 0.2, c= 6.5, init_time= 0., end_time= 32*np.pi, init= None):
        t = np.linspace(init_time, end_time, length)
        step_size = (end_time - init_time) / length
        x, y, z = [np.zeros_like(t) for _ in range(3)]
        if init is not None:
            x[0], y[0], z[0] = init
        for i in range(length-1):
            x[i+1] = x[i] + step_size * - (y[i] + z[i])
            y[i+1] = y[i] + step_size * (x[i] + a * y[i])
            z[i+1] = z[i] + step_size * (b + z[i] * (x[i] - c))
        return pd.DataFrame.from_dict({'t': t, 'x': x, 'y': y, 'z': z})

    def lorenz(ii, length=1, sigma=10, rho= 28, beta= 2.667, init=[0, 1, 1.05], init_time=0, end_time= 32*np.pi):

        step_size = (end_time - init_time) / length
        x, y, z = np.zeros((length,)), np.zeros((length,)), np.zeros((length,))
        if init is not None:
            x[0], y[0], z[0] = init
        for i in range(length-1):
            x[i+1] = x[i] + step_size * (sigma * (y[i] - x[i]))
            y[i+1] = y[i] + step_size * (x[i] * (rho - z[i]) - y[i])
            z[i+1] = z[i] + step_size * (x[i]*y[i] - beta*z[i])
        
        t = np.linspace(init_time, end_time, length)
        return pd.DataFrame.from_dict({'t': t, 'x': x, 'y': y, 'z': z})
    
    def all(ii, length= 1000, x_init= 0.0, **kw):
        d = {}
        d['quadratic_map'] = ii.quadratic_map(x_init, length= length, c= 1.2)
        d['mackey_glass'] = ii.mackey_glass(x_init= None, length= length, beta=0.2, gamma=0.1, tau= 17, n= 10)
        d['rossler'] = ii.rossler(length= length, a= 0.13, b= 0.2, c= 6.5, init_time=0, end_time=32*np.pi, init=None)
        d['lorenz'] = ii.lorenz(length= length, sigma= 10, rho=28, beta=2.667, init=[0, 1, 1.05], init_time= 0, end_time= 8*np.pi)
        return d