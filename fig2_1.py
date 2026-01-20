import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45,solve_ivp
import gc

tmax= 15000  #in years
pnts=100000 #no. of points

times=np.linspace(0,tmax,pnts) 

mx=100
mn=100
mr=100

η1=1
η2=1



def interaction(row, coloumn,  mu_1, sigma_1, mu_2, sigma_2, rho):  
                                                                    
                                                                    
                                                                    

    randompart = np.random.normal(0, 1, (row, coloumn))
    mat1 = (mu_1 ) + (sigma_1 * randompart    )

    mat2 = (mu_2 ) + (sigma_2  * (rho * randompart + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (row, coloumn))))

    return mat1, mat2
    
    
    
def lotkavoltera3d(t, y, C, D, E, F, m, u, k, add, mx, mn, η1, η2):
    """
    returns dx/dt, dn/dt, dr/dt
    Parameters
    -------------
    t = time evolved
    C = interaction matrix between producers and herbivores (consumer preferences for the species in bottom level
    D = interaction matrix between herbivores and carnivores  (ability to avoid predation by carnivores)
    m = death rate of herbivores(n)
    
    u = death rate of carnivore(x) 
    k = carrying capacity of the producers(r) 
    """

    x,n,r = y[:mx].T, y[mx:mx+mn].T, y[mn+mx:].T
    dxdt = x*(η1*(D.T@n) -  u)+add
    dndt = n*(η2*(C.T@r) - m - (E@x))+add
    drdt = r*(k-r) - r*(F@n)+add
    deriv = np.concatenate((dxdt.T,dndt.T,drdt.T))
    return(deriv)
    
mean_m=1
sd_m=0.1
m=np.random.normal(mean_m,(sd_m),mn)




mean_k=1
sd_k=0.1
k=np.random.normal(mean_k,(sd_k),mr)


mean_u=1
sd_u=0.1
u=np.random.normal(mean_u,(sd_u),mx)



mean_c = 100/mr
sd_c= 4/(np.sqrt(mr))
mean_f = 100/mr
sd_f= 4/(np.sqrt(mr))

mean_d = 100/mn
sd_d = 4/(np.sqrt(mn))
mean_e = 100/mn
sd_e = 4/(np.sqrt(mn))
add=1e-9

rho_r=np.round(np.arange(1,0-0.025,-0.05),2)
rho_x=np.round(np.arange(1,0-0.025,-0.05),2)


n=100#no_of_realisations
N=1000#how many last values to be saved
max_retries = 15  # Max attempts for a given (rho_x, rho_r) pair

for x in range(len(rho_x)):
    for r in range(len(rho_r)):
        
        for i in range(n):
            
            attempt = 0
            success = False
            while attempt < max_retries and not success:
                # generate initial conditions
                x0_arr = np.random.rand(mx)
                n0_arr = np.random.rand(mn)
                r0_arr = np.random.rand(mr)
                ini_val = np.concatenate((x0_arr, n0_arr, r0_arr))

                # generate interaction matrices (same rho values)
                D, E = interaction(mn, mx, mean_d, sd_d, mean_e, sd_e, rho_x[x])
                C, F = interaction(mr, mn, mean_c, sd_c, mean_f, sd_f, rho_r[r])
                params = (C, D, E, F, m, u, k, add, mx, mn, η1, η2)

                try:
                    sol = solve_ivp(
                        lotkavoltera3d,
                        t_span=(0, tmax),
                        y0=ini_val,
                        method="LSODA",
                        args=params,
                        t_eval=times,
                        dense_output=False,
                        rtol=1e-11,
                        atol=1e-10
                    )

                    if sol.success and not np.any(np.isnan(sol.y)):
                        # Save only if the solution is good
                        
                        sol_data = sol.y.copy()
                        del sol
                        gc.collect()
                        np.savez_compressed(
                            f'rhox={rho_x[x]}__rhor={rho_r[r]}__realisation={(100-n)+i}.npz',
                            c=sol_data[:mx, -N:],   
                            h=sol_data[mx:mx+mn, -N:],  
                            p=sol_data[mx+mn:, -N:]
                        )
                        np.savez_compressed(
                            f'entire__rhox={rho_x[x]}__rhor={rho_r[r]}__realisation={(100-n)+i}.npz',
                            c=sol_data[:mx, 99::100],   
                            h=sol_data[mx:mx+mn,99::100],  
                            p=sol_data[mx+mn:,99::100]
                        )
                        
                        success = True
                        print(f"Success on attempt {attempt+1} for realisation {(100-n)+i}, rho_x={rho_x[x]}, rho_r={rho_r[r]}")
                        del sol_data,x0_arr,n0_arr,r0_arr,ini_val,D,E,C,F,params
                        gc.collect()
                    else:
                        print(f"Retry {attempt+1} failed: Integration unsuccessful or NaNs detected.")
                        del sol,x0_arr,n0_arr,r0_arr,ini_val,D,E,C,F,params
                        gc.collect()
                except Exception as e:
                    print(f"Retry {attempt+1} raised error: {e}")

                attempt += 1

            if not success:
                print(f"❌ All {max_retries} retries failed for realisation {(100-n)+i}, rho_x={rho_x[x]}, rho_r={rho_r[r]}")
                
    
            








            
