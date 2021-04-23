2020/03/06
lateral mode v0's state space equation is probably have something wrong,
two out of five eigenvalues is 1000 times bigger than the given in the aircraft stability
slides. So, I decide to try another state-space equation in MAE 5070 Dynamics of Flight Vehicles, Sibley School of Mechanical and Aerospace Engineering

x =β p φ r  (6.151)
A =   
−0.0999 0.0000 0.1153 −1.0000 
−1.6038 −1.0932 0.0 0.2850 
0.0 1.0 0.0 0.0 
0.4089 −.0395 0.0 −.2454    

aileron-only control
B = [0.0000 0.3215 0.0000 −.0017]T    (6.163)
 rudder-only control
B = [0.0182 0.0868 0.0 −.2440]T (6.158) 
source https://courses.cit.cornell.edu/mae5070/Control.pdf
