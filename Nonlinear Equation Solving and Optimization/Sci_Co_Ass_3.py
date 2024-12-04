import numbers
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis, fill_diagonal, sum, sqrt
import math
from mpl_toolkits.mplot3d import Axes3D
NA = newaxis

# Experimental values for Argon atoms
EPSILON=0.997; # kJ/mol
SIGMA=  3.401; # Ångstrom

with np.load('ArStart.npz') as ArStart:
    Xstart2 = ArStart['Xstart2']
    Xstart3 = ArStart['Xstart3']
    Xstart4 = ArStart['Xstart4']
    Xstart5 = ArStart['Xstart5']
    Xstart6 = ArStart['Xstart6']
    Xstart7 = ArStart['Xstart7']
    Xstart8 = ArStart['Xstart8']
    Xstart9 = ArStart['Xstart9']
    Xstart20 = ArStart['Xstart20']
    Xopt20 = ArStart['Xopt20']
Xstarts =[]
Xstarts.append(Xstart2);Xstarts.append(Xstart3);Xstarts.append(Xstart4);Xstarts.append(Xstart5);Xstarts.append(Xstart6);Xstarts.append(Xstart7);Xstarts.append(Xstart8);Xstarts.append(Xstart9);Xstarts.append(Xstart20)

def distance(points):
    # points: (N,3)-array of (x,y,z) coordinates for N points
    # distance(points): returns (N,N)-array of inter-point distances:
    displacement = points[:, NA] - points[NA, :]
    return sqrt(sum(displacement*displacement, axis=-1))


def LJ(sigma=SIGMA, epsilon=EPSILON):
    def V(points):
        # points: (N,3)-array of (x,y,z) coordinates for N points
        dist = distance(points)

        # Fill diagonal with 1, so we don't divide by zero
        fill_diagonal(dist, 1)

        # dimensionless reciprocal distance
        f = sigma/dist

        # calculate the interatomic potentials
        pot = 4*epsilon*(f**12 - f**6)

        # Undo any diagonal terms (the particles don't interact with themselves)
        fill_diagonal(pot, 0)

        # We could return the sum of just the upper triangular part, corresponding
        # to equation 2 in the assignment text. But since the matrix is symmetric,
        # we can just sum over all elements and divide by 2
        return sum(pot)/2
    return V


def LJgradient(sigma=SIGMA, epsilon=EPSILON):
    def gradV(X):
        d = X[:, NA] - X[NA, :]
        r = sqrt(sum(d*d, axis=-1))

        fill_diagonal(r, 1)

        T = 6*(sigma**6)*(r**-7)-12*(sigma**12)*(r**-13)
        # (N,N)−matrix of r−derivatives
        # Using the chain rule , we turn the (N,N)−matrix of r−derivatives into
        # the (N,3)−array of derivatives to Cartesian coordinate: the gradient.
        # (Automatically sets diagonal to (0,0,0) = X[ i]−X[ i ])
        u = d/r[:, :, NA]
        # u is (N,N,3)−array of unit vectors in directi
        # on of X[ i ]−X[ j ]
        return 4*epsilon*sum(T[:, :, NA]*u, axis=1)
    return gradV


# wrapper functions to generate "flattened" versions of the potential and gradient.
def flatten_function(f):
    return lambda x: f(x.reshape(-1, 3))


def flatten_gradient(f):
    return lambda x: f(x.reshape(-1, 3)).reshape(-1)

# potential and gradient with values for Argon
V = LJ()
gradV = LJgradient()

# Flattened potential and gradient.
flat_V     = flatten_function(V)
flat_gradV = flatten_gradient(gradV)

def bisection_root_method(calls,f,f_a,f_b,a,b,tolerance = 1e-13):#I have extra inputs because I use recursion 
    x_midpoint = bisection_root(f_a,f_b,a,b) #calculate the center of the area 
    f_midpoint = f(x_midpoint) #calculate the functions value in the center
    calls +=1 #thats the only time I call f
    if np.abs(f_midpoint) < tolerance:
        return (a+b)/2, calls

    if  f_midpoint > 0:  # Now I choose either I will change a or b
        a = (a+b)/2
        f_a = f_midpoint 
    elif f_midpoint <0:
        b = (a+b)/2
        f_b = f_midpoint
    return bisection_root_method(calls,f,f_a,f_b,a,b,tolerance)

def bisection_root_method_3N(calls,f,f_a,f_b,a,b,d,tolerance = 1e-10):
    x_new = bisection_root(f_a,f_b,a,b)
    key = np.dot(f(x_new).flatten(),d.flatten())
    calls +=1
    if np.abs(key) < tolerance:
        return (a+b)/2, calls
    if  key <0:
        a = (a+b)/2
        f_a = key
    elif key >0:
        b = (a+b)/2
        f_b = key
    return bisection_root_method_3N(calls,f,f_a,f_b,a,b,d,tolerance)

def bisection_root(f_a,f_b,a,b):
    if f_a<f_b:
       temp = a
       a = b
       b=temp
    return a+((b-a)/2)

def potential_energy(x):
    x1 = [0,0,0]
    x= [x,0,0]
    points = np.vstack((x,x1))
    potential = V(points)
    return potential

def four_particles_potential_energy(x):
    x= [x,0,0]
    x1 = [0,0,0]
    x2 = [14,0,0]
    x3 = [7,3.2,0]
    points = np.vstack((x,x1,x2,x3))
    potential = V(points)
    return potential

def grad_potential_energy(x):
    x1 = [0,0,0]
    x= [x,0,0]
    points = np.vstack((x,x1))
    grad_potential = gradV(points)
    return grad_potential

def newton_root(f,df,x0,tolerance,max_iterations):
    f_x0 = f(x0) #calculate how close to 0 we are when we start
    call =1
    for i in range(max_iterations):
        x_new= newton_new_point(f_x0,df,x0) #f_x0 was calculated in the previous iteration of the loop
        f_x0 = f(x_new) 
        call+=2 # 1 time inside newton for the gradient and on in the line above
        if f_x0<tolerance:
            return x_new, call
        x0 = x_new
    return x0 , call

def newton_new_point(f_x0,df,x0):
    return x0 + f_x0/df(x0)[1,0]

def golden_section_min(f,a,b):
    #apply golden search in between [a,b]
    calls = 0
    tau = (sqrt(5)-1)/2
    tolerance=0.001
    x1 = a+(1-tau)*(b-a)
    x2 = a+tau*(b-a)
    f1 = f(x1)
    f2 = f(x2)
    calls+=2
    while np.abs(b-a) > tolerance:
        if f1>f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a+tau*(b-a)
            f2 = f(x2)
            calls+=1
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1-tau)*(b-a)
            calls+=1
            f1 = f(x1)
    return (b+a)/2, calls

def line_function(F, x0, d):
    """
    Returns a function, F_restricted, which calculates F along the line defined
    by x0+alpha*d.
    """
    def F_restricted(alpha):
        # A line through space
        line = x0 + alpha*d
        # Calculate F along the line
        return F(line)
    return F_restricted

def linesearch(f,x0,d,alpha_max,tolerancem, max_iteratins):
    f_along_line = line_function(f, x0, d)
    calls=0
    f_a = np.dot(f_along_line(0).flatten(),d.flatten())
    f_b = np.dot(f_along_line(1).flatten(),d.flatten())
    alpha, calls = bisection_root_method_3N(calls,f_along_line,f_a,f_b,0,1,d)
    return alpha, calls

def combined_methods(a,b,x0,tolerance):
    v_a = potential_energy(a)
    v_b = potential_energy(b)
    f_x0 = potential_energy(x0)
    total_calls = 3 #the counter starts from 3 because of the 3 calculations above

    for i in range(1000):
        x = newton_new_point(f_x0,grad_potential_energy,x0) 
        total_calls+=1
        if x>b or x<a: #check if the newton's method ouput a value outside my confined area
            x0 = bisection_root(v_a,v_b,a,b)
            v_x0 = potential_energy(x0)
            total_calls+=1
            if np.sign(v_x0) == np.sign(v_a):
                a = x0
                v_a = v_x0
                x = a
            elif np.sign(v_x0) == 0:
                return x0,total_calls
            else:
                x = x0
                b = x0
                v_b = v_x0
        x0=x
        total_calls+=1 #this is because of the calculation in the next line
        f_x0 = potential_energy(x0)
        if np.abs(f_x0)<1e-13:
            return x0, total_calls
    return x, total_calls

def calculate_distances(x0, number_of_particles):
    d = np.zeros([number_of_particles,number_of_particles])
    for i in range(number_of_particles):
        for j in range(number_of_particles):
            d[i,j] = math.dist(x0[3*i:3*i+3],x0[3*j:3*j+3])
    print('----------')
    print("for ",number_of_particles," particles system we have ",(sum(abs(d-3.81)/3.81<=0.02)//2)," distances with less than 1% error of the optimal")
    return d

def n_particle_minimum(with_linesearch,which_graphs_to_show):
    #this function calculates the distances, makes the plots, and outputs on the terminal
    for x in Xstarts:
        x0, k, converged = BFGS(flat_V, flat_gradV, x, with_linesearch)
        number_of_particles = int(np.size(x0)/3)
        d = calculate_distances(x0,number_of_particles) #d is the distance matrix
        if converged == True:      
            print("Converged = ",converged)
            coordinates=np.zeros([number_of_particles,3])
            for i in range(number_of_particles): # in this line I make the flattened array to a matrix in the most complex way possible
                coordinates[i]= x0[i*3:i*3+3]
            if number_of_particles in which_graphs_to_show: #here i choose to plot only the ones that I show on the report
                fig4 = plt.figure(number_of_particles+10)
                ax = fig4.gca(projection='3d')
                ax.set_xlabel('$X$', fontsize=20)
                ax.set_ylabel('$Y$', fontsize=20)
                ax.set_zlabel('$Z$', fontsize=20, rotation = 0)
                if with_linesearch == True:
                    string = "with"
                else:
                    string = "without"
                ax.scatter3D(coordinates[:,0],coordinates[:,1],coordinates[:,2], c='r', marker='o')
                
                string = str(number_of_particles)+" particles "+string +" lineasearch"
                ax.set_title(str(string))
            print("Iterations =", k)
            
        else:
            return

def BFGS(f, gradf, X, with_linesearch,tolerance=1e-6,max_iterations=10000):
    x = X.copy()
    dfx = gradf(x)
    i = 0
    n_reset = 100
    converged = False
    for k in range(max_iterations):
        if i % n_reset == 0:#every 100 iteration we reset B to identity
            B = np.eye(len(x)) # starts with steepest descent
        i += 1

        if with_linesearch:
            alpha, a_k = golden_section_min(lambda a: f(x-a*gradf(x)), 0, 1)
            dx = - alpha * B @ dfx
            k += a_k
        else:
            dx = -B @ dfx
        x += dx
        grad =  gradf(x)
        dy = grad - dfx
        dfx = grad
        denom = np.dot(dx,dy)
        nx = np.sqrt(np.dot(dx,dx))
        ny = np.sqrt(np.dot(dy,dy))
        if(denom/(nx*ny) < 1e-10): #we want to avoid overflow when updating the inverse hessian when   y and s are orthogonal
            denom = nx*ny
        B += np.outer(dx, dx) / denom - B @ ((np.outer(dy, dy) @ B) / (dy.T @ B @ dy))
        if np.linalg.norm(dfx) < tolerance: #return the result and the calls when we are very close to the solution
            converged = True
            return x, k+1, converged #the +1 on k is because i called gradf ones out of the loop
    return x, k+1, converged

def question_a():
    print("_____________________Question a________________________")
    print("Two wild graphs should appear")
    n=100
    x0 = np.linspace(3,11,n)
    potentials = np.zeros([n])
    i=0
    plot1 = plt.figure(1)
    for x in x0:
        potentials[i] = potential_energy(x)
        i+=1
    
    plt.style.use('seaborn-whitegrid')
    plt.plot( x0, potentials)
    plt.xlabel('Distance')
    plt.ylabel('Potential Energy')
    plt.savefig('2_particlesPotential')

    i=0
    plot2 = plt.figure(2)
    for x in x0:
        potentials[i] = four_particles_potential_energy(x)
        i+=1
        
    plt.plot( x0, potentials)
    plt.xlabel('Distance')
    plt.ylabel('Potential Energy')    
    plt.style.use('seaborn-whitegrid')
    plt.savefig('4_particlesPotential')

def question_b():
    print("__________Question b-Bisection Method________________________")
    calls=0
    x, calls = bisection_root_method(calls,potential_energy,potential_energy(2),potential_energy(6),2,6)
    print("x = ",x)
    print("calls = ",calls+2) #the +2 is because I call the potential energy function 2 extra times in inputs

def question_c():
    print("__________Question c-Newton's Method________________________")
    x, calls = newton_root(potential_energy,grad_potential_energy,2,1e-12,1000)
    print("x = ",x)
    print("calls = ",calls)

def question_d():
    print("__________Question d-Combined Methods________________________")
    a=2
    b=6
    x0 = 2
    x,calls = combined_methods(a,b,x0,1e-13)

    print('Starting bracket [a,b] =[',a,',',b,']')
    print('Starting approximation x0=',x0)
    print("x = ",x)
    print("calls = ",calls)

    a=2
    b=6
    x0 = 5.5
    x,calls = combined_methods(a,b,x0,1e-13)

    print('\nStarting bracket [a,b] =[',a,',',b,']')
    print('Starting approximation x0=',x0)
    print("x = ",x)
    print("calls = ",calls)

def question_e():
    print("________________________Question e________________________")

    n=10
    x0 = np.zeros([n,3])
    x0[:,0] = np.linspace(3,11,n)
    x0
    x1 = [0,0,0]
    points = np.vstack((x0,x1))
    print(gradV(points))



def question_f():
    print("________________________Question f________________________")

    x0 = [[4,0,0],[0,0,0],[14,0,0],[7,3.2,0]]
    x0 = np.array(x0)
    d = -gradV(x0)
    alpha, calls = linesearch(gradV, x0,d,1,1e-13,1000)
    print('a = ',alpha,', and the calculatin was made with ',calls,' calls of the functions')

def question_g():
    print("________________________Question g________________________")
    print("____________1___________")
    x0 = [[4,0,0],[0,0,0],[14,0,0],[7,3.2,0]]
    x0 = np.array(x0)
    d = -gradV(x0)
    f_along_line = line_function(V, x0, d)
    calls=0
    x, calls = golden_section_min(f_along_line,0,1)
    print('alpha was calculated ',x,' with ',calls,' calls')
    print("____________2___________")
    v_a = potential_energy(3)
    v_b = potential_energy(2)
    x, calls = golden_section_min(potential_energy,0,5)
    print('the distance between two Ar atoms was calculated ',x,' with ',calls,' calls')
    return

def question_h():
    print("__________________________________Question h__________________________________")
    x0, k, converged = BFGS(flat_V, flat_gradV, Xstart2, False)
    print("Converged = ",converged)
    if converged == True:
        print("Distance of conversion = ", math.dist(x0[0:3],x0[3:6]))
        print("Iterations =", k)


def question_i():
    print("__________________________________Question i__________________________________")
    which_graphs_to_show = [3,8]
    n_particle_minimum(False,which_graphs_to_show)#the input dictates the usage of lineasearch
    return


def question_j():
    print("__________________________________Question j_________________________________")
    which_graphs_to_show = [9,20]
    n_particle_minimum(True,which_graphs_to_show)#the first input dictates the usage of lineasearch
    return

#CODE EXECUTION

#Week 1
question_a()
question_b()
question_c()
question_d()
question_e()
question_f()
#Week 2
question_g()
question_h()
question_i()
question_j()


plt.show()