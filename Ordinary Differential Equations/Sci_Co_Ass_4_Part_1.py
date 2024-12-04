import numpy as np
import matplotlib.pyplot as plt

#COMMENTS ON THE SCALING OF THE POPULATION
#I realised that if I want to scale the population I should also scale the parameters of infection.
#the reason is that there is a multiplication between the populations so i get a square of the scaling of each population.
#But I want to get the same result if I scale the total populations and the initial infected
#My to choices were either to scale down the infection rate or scale down the time scale.
#If I scale down the time scale I should also scale down the parameters for transfusion and death rate so it becomes complicated
#so thats why chose to just scale down the infection parameters as can be seen bellow where I devide by the population.
population = 10 
a1              = 10/population #the rate in which HEALTHY HOMOSEXUAL MALES are infected by HOMOSEXUAL MALES
a2              = 5/population     #the rate in which HEALTHY HOMOSEXUAL MALES are infected by BISEXUAL MALES

b1              = 5/population     #the rate in which HEALTHY BISEXUAL MALES are infected by HOMOSEXUAL MALES
b2              = 1/population     #the rate in which HEALTHY BISEXUAL MALES are infected by BISEXUAL MALES
b3              = 1/population     #the rate in which HEALTHY BISEXUAL MALES are infected by HETEROSEXUAL FEMALES

c1              = 1/population     #the rate in which HEALTHY HETEROSEXUAL MALES are infected by BISEXUAL MALES
c2              = 1/population     #the rate in which HEALTHY HETEROSEXUAL MALES are infected by HETEROSEXUAL FEMALES

p1              = 5*population     #p1 homosexuals that x1(t) are infected
p2              = 5*population     #bisexuals that x2(t) are infected
q               = 100*population   #q heterosexual females that y(t) are infected
r               = 100*population   #r heterosexual males that z(t) are infected


x1              = 0.01*population  #initial infected homosexual males
x2              = 0*population  #initial infected bisexuals males
y               = 0*population  #initial heterosexual females
z               = 0*population  #initial infected heterosexual males

h               = 0.001 #step
d,d1            = 1/population,1/population
#this function calculated the parameter for the blood transfusion
#when the transfusion_parameter is 0 it is like I have no transfussion
def calculate_e(x1,x2,y,z,p1,q,r,tranfusion_parameter):
    return tranfusion_parameter*(x1+x2+y+z)/(p1+p2+q+r)

#Calculation of the derivatives
def dx1(x1,x2,a1,a2,p1,e,r1):
    return a1*x1*(p1-x1)+a2*x2*(p1-x1)+e*(p1-x1)-r1*x1

def dx2(x1,x2,b1,b2,b3,p2,e,r2,y):
    return b1*x1*(p2-x2)+b2*x2*(p2-x2)+b3*y*(p2-x2)+e*(p2-x2)-r2*x2

def dy(x2,c1,c2,z,q,y,e,r3):
    return (c1*x2+c2*z)*(q-y)+e*(q-y)-r3*y

def dz(d1,y,r,z,e,r4):
    return d1*y*(r-z)+e*(r-z)-r4*z

#backwards-euler method
def euler_implementation(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter,include_population_reduction):
    dt = h
    x1_euler, x2_euler, y_euler, z_euler = np.zeros([iters]),np.zeros([iters]),np.zeros([iters]),np.zeros([iters])
    p1_euler, p2_euler, q_euler, r_euler = np.zeros([iters]),np.zeros([iters]),np.zeros([iters]),np.zeros([iters])

    x1_euler[0] = x1
    x2_euler[0] = x2
    y_euler[0] = y
    z_euler[0] = z
    p1_euler[0]=p1
    p2_euler[0] = p2
    q_euler[0]=q
    r_euler[0]=r

    for i in range(iters-1):
        x1 = x1_euler[i]
        x2 = x2_euler[i]
        y = y_euler[i]
        z = z_euler[i]
        if include_population_reduction==1: #here I chech if i should take into consideration the reduction of the population
            p1= p1_euler[i]
            p2= p2_euler[i]
            q= q_euler[i]
            r= r_euler[i]
        e = calculate_e(x1,x2,y,z,p1,q,r,tranfusion_parameter )
        x1_euler[i+1] = x1_euler[i] + dt* dx1(x1_euler[i],x2_euler[i],a1,a2,p1,e,r1)
        x2_euler[i+1] = x2_euler[i] + dt* dx2(x1_euler[i],x2_euler[i],b1,b2,b3,p2,e,r2,y)
        y_euler[i+1] = y_euler[i] + dt* dy(x2_euler[i],c1,c2, z_euler[i],q,y_euler[i],e,r3)
        z_euler[i+1] = z_euler[i] + dt* dz(d1,y_euler[i],r,z_euler[i],e,r4)
        if include_population_reduction==1: #here I chech if i should take into consideration the reduction of the population
            p1_euler[i+1] = p1_euler[i]-h*r1*x1
            p2_euler[i+1] =p2_euler[i]-h*r2*x2
            q_euler[i+1]  =q_euler[i] -h*r3*y
            r_euler[i+1]  =r_euler[i] -h*r4*z
    return np.array([x1_euler, x2_euler, y_euler, z_euler,p1_euler,p2_euler,q_euler,r_euler])

#this function calculates all the runge-kutta parameters
def differential_eq(x1,x2,y,z,a1,a2,b1,b2,b3,c1,c2,d1,p1,p2,q,r,r1,r2,r3,r4,e,h,k,l,m,n):
    k = dx1(x1+h*k,x2+h*l,a1,a2,p1,e,r1)
    l = dx2(x1+h*k,x2+h*l,b1,b2,b3,p2,e,r2,y)
    m = dy(x2+h*l,c1,c2,z+h*n,q,y+h*m,e,r3)
    n = dz(d1,y+h*l,r,z+h*n,e,r4)
    return k,l,m,n
#runge-kutta 4 method
def rk4_implementation(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter,include_population_reduction ):
    x1_rk4, x2_rk4, y_rk4, z_rk4 = np.zeros([iters]),np.zeros([iters]),np.zeros([iters]),np.zeros([iters])
    p1_rk4, p2_rk4, q_rk4, r_rk4 = np.zeros([iters]),np.zeros([iters]),np.zeros([iters]),np.zeros([iters])
    x1_rk4[0] = x1
    x2_rk4[0] = x2
    y_rk4[0] = y
    z_rk4[0] = z
    p1_rk4[0]=p1
    p2_rk4[0] = p2
    q_rk4[0]=q
    r_rk4[0]=r
    for i in range(iters-1):
        x1 = x1_rk4[i]
        x2 = x2_rk4[i]
        y = y_rk4[i]
        z = z_rk4[i]
        if include_population_reduction==1: #here I chech if i should take into consideration the reduction of the population
            p1= p1_rk4[i]
            p2= p2_rk4[i]
            q= q_rk4[i]
            r= r_rk4[i]
        e = calculate_e(x1,x2,y,z,p1,q,r,tranfusion_parameter )

        k1,l1,m1,n1 = differential_eq(x1,x2,y,z,a1,a2,b1,b2,b3,c1,c2,d1,p1,p2,q,r,r1,r2,r3,r4,e,0,0,0,0,0)
        k2,l2,m2,n2 = differential_eq(x1,x2,y,z,a1,a2,b1,b2,b3,c1,c2,d1,p1,p2,q,r,r1,r2,r3,r4,e,0.5*h,k1,l1,m1,n1)
        k3,l3,m3,n3 = differential_eq(x1,x2,y,z,a1,a2,b1,b2,b3,c1,c2,d1,p1,p2,q,r,r1,r2,r3,r4,e,0.5*h,k2,l2,m2,n2)
        k4,l4,m4,n4 = differential_eq(x1,x2,y,z,a1,a2,b1,b2,b3,c1,c2,d1,p1,p2,q,r,r1,r2,r3,r4,e,1*h,k3,l3,m3,n3)

        x1_rk4[i+1]= x1_rk4[i]+h*(k1+2*k2+2*k3+k4)/6
        x2_rk4[i+1]= x2_rk4[i]+h*(l1+2*l2+2*l3+l4)/6
        y_rk4[i+1]= y_rk4[i]+h*(m1+2*m2+2*m3+m4)/6
        z_rk4[i+1]= z_rk4[i]+h*(n1+2*n2+2*n3+n4)/6
        if include_population_reduction==1: #here I chech if i should take into consideration the reduction of the population
            p1_rk4[i+1] = p1-h*r1*x1
            p2_rk4[i+1] =p2-h*r2*x2
            q_rk4[i+1]  =q -h*r3*y
            r_rk4[i+1]  =r -h*r4*z
    return np.array([x1_rk4, x2_rk4, y_rk4, z_rk4,p1_rk4,p2_rk4,q_rk4,r_rk4])

#every time I run a simulation I apply both methods. This function just calls both methods and outputs all the data created
def implement_both(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter,include_population_reduction=0):
    data = np.zeros([2,8,iters])
    data[0] = euler_implementation(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter,include_population_reduction)
    data[1] = rk4_implementation(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter,include_population_reduction)
    return data

#THE REST OF THE FUNCTIONS ARE JUST PLOTTING
#this function plots the 2 methods for only 1 model
def plot_1(result,iters):
    fig3, ax3= plt.subplots(1,1)
    iters = range(iters)

    ax3.plot(iters,result[1,0], c = 'r',label = 'x1, infected homosexuals'  )
    ax3.plot(iters,result[1,1], c = 'b', label = 'x2, infected bisexuals'   )
    ax3.plot(iters,result[1,2], c = 'g',label = 'y, heterosexual females'   )
    ax3.plot(iters,result[1,3], c = 'y', label ='z, heterosexual males'     )  
    ax3.set_title('rk4 method')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.grid()
    ax3.grid();ax3.grid()
    ax3.legend(bbox_to_anchor=(1,1),loc = "upper left")
    
#this function plots the 2 methods for 2 models

def plot_2(result,iters):
    fig2, ax2= plt.subplots(1,2)
    iters = range(iters)
    ax2[0].plot(iters,result[0,0,0], c = 'r',label = 'x1, infected homosexuals')
    ax2[0].plot(iters,result[0,0,1], c = 'b', label = 'x2, infected bisexuals')
    ax2[0].plot(iters,result[0,0,2], c = 'g',label = 'y, heterosexual females')
    ax2[0].plot(iters,result[0,0,3], c = 'y', label ='z, heterosexual males') 
    ax2[0].plot(iters,result[1,0,0], c = 'r', linestyle ="dotted")
    ax2[0].plot(iters,result[1,0,1], c = 'b', linestyle ="dotted")
    ax2[0].plot(iters,result[1,0,2], c = 'g', linestyle ="dotted")
    ax2[0].plot(iters,result[1,0,3], c = 'y', linestyle ="dotted") 
    ax2[0].set_title('euler method')
    plt.grid()

    ax2[1].plot(iters,result[0,1,0], c = 'r',label = 'x1, infected homosexuals')
    ax2[1].plot(iters,result[0,1,1], c = 'b', label = 'x2, infected bisexuals')
    ax2[1].plot(iters,result[0,1,2], c = 'g',label = 'y, heterosexual females')
    ax2[1].plot(iters,result[0,1,3], c = 'y', label ='z, heterosexual males')  
    ax2[1].plot(iters,result[1,0,0], c = 'r', linestyle ="dotted")
    ax2[1].plot(iters,result[1,0,1], c = 'b', linestyle ="dotted")
    ax2[1].plot(iters,result[1,0,2], c = 'g', linestyle ="dotted")
    ax2[1].plot(iters,result[1,0,3], c = 'y', linestyle ="dotted") 
    ax2[1].set_title('rk4 method')
    ax2[0].grid();ax2[1].grid()
    ax2[1].legend(bbox_to_anchor=(0.25,0.25),loc = "center right")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.grid()

#this function plots the 2 methods for 3 models
def plot_3(result,iters):
    fig2, ax2= plt.subplots(1,2)
    iters = range(iters)
    ax2[0].plot(iters,result[0,0,0], c = 'r',label = 'x1, infected homosexuals')
    ax2[0].plot(iters,result[0,0,1], c = 'b', label = 'x2, infected bisexuals')
    ax2[0].plot(iters,result[0,0,2], c = 'g',label = 'y, heterosexual females')
    ax2[0].plot(iters,result[0,0,3], c = 'y', label ='z, heterosexual males') 
    ax2[0].plot(iters,result[1,0,0], c = 'r', linestyle ="dotted")
    ax2[0].plot(iters,result[1,0,1], c = 'b', linestyle ="dotted")
    ax2[0].plot(iters,result[1,0,2], c = 'g', linestyle ="dotted")
    ax2[0].plot(iters,result[1,0,3], c = 'y', linestyle ="dotted") 
    ax2[0].plot(iters,result[2,0,0], c = 'r', linestyle ="dashdot")
    ax2[0].plot(iters,result[2,0,1], c = 'b', linestyle ="dashdot")
    ax2[0].plot(iters,result[2,0,2], c = 'g', linestyle ="dashdot")
    ax2[0].plot(iters,result[2,0,3], c = 'y', linestyle ="dashdot") 
    ax2[0].set_title('euler method')
    plt.grid()

    ax2[1].plot(iters,result[0,1,0], c = 'r',label = 'x1, infected homosexuals')
    ax2[1].plot(iters,result[0,1,1], c = 'b', label = 'x2, infected bisexuals')
    ax2[1].plot(iters,result[0,1,2], c = 'g',label = 'y, heterosexual females')
    ax2[1].plot(iters,result[0,1,3], c = 'y', label ='z, heterosexual males')  
    ax2[1].plot(iters,result[1,1,0], c = 'r', linestyle ="dotted")
    ax2[1].plot(iters,result[1,1,1], c = 'b', linestyle ="dotted")
    ax2[1].plot(iters,result[1,1,2], c = 'g', linestyle ="dotted")
    ax2[1].plot(iters,result[1,1,3], c = 'y', linestyle ="dotted") 
    ax2[1].plot(iters,result[2,1,0], c = 'r', linestyle ="dashdot")
    ax2[1].plot(iters,result[2,1,1], c = 'b', linestyle ="dashdot")
    ax2[1].plot(iters,result[2,1,2], c = 'g', linestyle ="dashdot")
    ax2[1].plot(iters,result[2,1,3], c = 'y', linestyle ="dashdot") 
    ax2[1].set_title('rk4 method')
    ax2[0].grid();ax2[1].grid()
    ax2[1].legend(bbox_to_anchor=(0.25,0.25),loc = "center right")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.grid()
    
#CODE EXECUTION
#All the parameters for the blood transfussion and the death rates don;t play a role when they are zero
#No deaths, no blood transfusion
tranfusion_parameter = 0
iters = 400
results = np.zeros([3,2,8,iters])

results[0] = implement_both(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,0,0,0,0,h,iters,tranfusion_parameter)
#No deaths, yes blood transfusion
tranfusion_parameter  = 20
results[1] = implement_both(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,0,0,0,0,h,iters,tranfusion_parameter)

#Everything
r_              = 10    #death rate. Increase slowly
r1,r2,r3,r4     = r_,r_,r_,r_
results[2] = implement_both(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter)

plot_1(results[0,:,:5,:200],200)
plot_2(results[0:2,:,:,:200],200)
plot_3(results[:,:,:5,:200],200)

#Everything with population reduction
include_population_reduction=1
r_              = 40    #death rate. Increase slowly
r1,r2,r3,r4     = r_,r_,r_,r_
result = implement_both(a1,a2,b1,b2,b3,c1,c2,p1,p2,q,r,x1,x2,y,z,r1,r2,r3,r4,h,iters,tranfusion_parameter,include_population_reduction)
plot_2(np.stack((result[:,0:4,:iters],result[:,4:8,:iters])),iters)

plt.show()