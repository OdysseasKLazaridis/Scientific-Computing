import numpy as np
import math
import matplotlib.pyplot as plt
#Functions
#region Functions

def ReverseMatrix(M):
    result = np.linalg.inv(M)
    return result

def max_norm(M):
    return np.max(np.sum( np.abs(M),axis=1))

def condition_number(M):
    return max_norm(M)*max_norm(ReverseMatrix(M))

def bounds_and_condition_numbers():
    conditionNumbers = np.array([0,0,0])
    bounds = np.zeros(3)
    for i in range(3):
        conditionNumbers[i]=condition_number(matrix[i,:,:])
        bounds[i] = conditionNumbers[i]*max_norm(5e-4*S)/max_norm(matrixCalculator(omega[i]))
    return conditionNumbers, bounds

def matrixCalculator(omega):
    return E - omega* S

def lu_factorize(M):
    L = np.identity(len(M))
    U = (M.copy()).astype(float)
    
    for j in range(len(M)-1):
        for i in range(j,len(M)-1):
            multiplier = U[i+1,j]/U[j,j]
            U[i+1,:] = U[i+1,:] -(multiplier*U[j,:])
            L[i+1,j] = multiplier
    return L, U

def forward_substitute(L,y):
    x = np.zeros(len(y))
    x[0] = y[0]/L[0,0]
    for i in range(1,len(y)):
        if(i!=0):
            sum = 0
            for j in range(i):
                sum += L[i,j]*x[j]
            x[i] = (y[i] - sum)/L[i,i]
    return x

def back_substitute(L,y):
    y = y[:len(L)]
    L,y = problem_flipper(L,y)
    x = np.flipud(forward_substitute(L,y))#the result is flipped because the forward_substitute gives the solution upside down
    return x

def problem_flipper(L,y): #Transforms the back_substitute system into forward_substitute system
    _U = np.flipud(np.fliplr(L))
    _y = np.flipud(y)
    return _U, _y

def solve_alpha(omega):
    matrix = matrixCalculator(omega)
    L, U = lu_factorize(matrix)
    y = forward_substitute(L,z)
    x = back_substitute(U,y)
    return np.dot(np.transpose(z),x)

def array_to_vector(x):
    x.shape = (1, x.shape[0])
    return x

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

def reflection_vector(a):
    e1 = np.zeros_like(a)
    e1[0, 0] = 1
    return norm(a) * e1
    
def householder_transformation(v):
    ref_vector = reflection_vector(v)
    u = (v + ref_vector*np.sign(v[0,0])).astype(np.float32)
    H = np.identity( v.shape[1]) - ((2 * np.matmul(np.transpose(u), u)) / np.matmul(u, np.transpose(u)))
    return H, u

def qr_factorization_slow(Q, R, i, n, m):
    a = array_to_vector(R[i:, i])
    Hbar, v = householder_transformation(a)
    H = np.identity(n)
    H[i:, i:] = Hbar
    R = H@R
    Q = Q@H
    return Q, R

def householder_QR_slow(A):
    R = A.astype(np.float32)
    n,m = np.shape(A)
    Q = np.identity(n)
    for i in range(min(n, m)):
        Q, R = qr_factorization_slow(Q, R, i, n, m)
    return Q, R

def qr_factorization_fast(R, i, n, m):
    a = array_to_vector(R[i:, i])
    Hbar, v = householder_transformation(a)
    H = np.identity(n)
    H[i:, i:] = Hbar
    R = H@R
    return R, v

def householder_fast(A):
    R = A.astype(np.float32)
    n,m = np.shape(A)
    VR=np.zeros([n+1,m])
    v=np.zeros([n,m])
    for i in range(min(n, m)):
        R, v[i:,i] = qr_factorization_fast(R, i, n, m)
    VR[:n, :] = R
    for i in range(min(n, m)):
        VR[i+1:, i] =v[i:,i]
    return VR

def least_squares(A,b):
    Q, R = householder_QR_slow(A)
    Q_T = Q.transpose()
    b = Q_T@b
    m,n = np.shape(A)
    minimum = min(m,n)
    b_ = b[:minimum]
    R_ = R[:minimum,:minimum]
    r=0
    for i in range(len(A),len(b)):
        r+=b[i]**2
    return back_substitute(R_,b_), r

def polynomial_approximation(omegas, grade):
    np.stack(omegas)
    alphas = (solve_alpha_vect(omegas))
    omegas_p = np.ones([1,len(omegas)])
    for i in range(1, grade+1):
        omegas_p = np.block([[omegas_p], [omegas**(2*i)]])
    omegas_p = np.transpose(np.array(omegas_p))

    return least_squares(omegas_p,alphas)

def fractional_polynomial_approximation(omegas, grade):
    np.stack(omegas)
    alphas = (solve_alpha_vect(omegas))
    omegas_p = np.ones([1,len(omegas)])
    for i in range(2):
        for j in range(1,grade+1):
            omegas_p = np.block([[omegas_p], [(omegas**j)*(alphas**i)]])
    omegas_p = np.transpose(np.array(omegas_p))
    return least_squares(omegas_p,alphas)
#endregion

Amat = np.array([
    [22.13831203, 0.16279204, 0.02353879, 0.02507880,-0.02243145,-0.02951967,-0.02401863],
    [0.16279204, 29.41831006, 0.02191543,-0.06341569, 0.02192010, 0.03284020, 0.03014052],
    [0.02353879,  0.02191543, 1.60947260,-0.01788177, 0.07075279, 0.03659182, 0.06105488],
    [0.02507880, -0.06341569,-0.01788177, 9.36187184,-0.07751218, 0.00541094,-0.10660903],
    [-0.02243145, 0.02192010, 0.07075279,-0.07751218, 0.71033323, 0.10958126, 0.12061597],
    [-0.02951967, 0.03284020, 0.03659182, 0.00541094, 0.10958126, 8.38326265, 0.06673979],
    [-0.02401863, 0.03014052, 0.06105488,-0.10660903, 0.12061597, 0.06673979, 1.15733569]]);

Bmat = np.array([
    [-0.03423002, 0.09822473,-0.00832308,-0.02524951,-0.00015116, 0.05321264, 0.01834117],
    [ 0.09822473,-0.51929354,-0.02050445, 0.10769768,-0.02394699,-0.04550922,-0.02907560],
    [-0.00832308,-0.02050445,-0.11285991, 0.04843759,-0.06732213,-0.08106876,-0.13042524],
    [-0.02524951, 0.10769768, 0.04843759,-0.10760461, 0.09008724, 0.05284246, 0.10728227],
    [-0.00015116,-0.02394699,-0.06732213, 0.09008724,-0.07596617,-0.02290627,-0.12421902],
    [ 0.05321264,-0.04550922,-0.08106876, 0.05284246,-0.02290627,-0.07399581,-0.07509467],
    [ 0.01834117,-0.02907560,-0.13042524, 0.10728227,-0.12421902,-0.07509467,-0.16777868]]);

yvec= np.array([-0.05677315,-0.00902581, 0.16002152, 0.07001784, 0.67801388,-0.10904168, 0.90505180]);
ones = np.ones([14,1])
deltaY = ones*(5e-9)
omega = np.array([0.800, 1.146, 1.400])

A = np.array([[12 , -51 , 4], [6 ,167,-68],[-4,24,-41]])
E = np.block([[Amat,Bmat],[Bmat,Amat]])
S = np.block([[np.identity(7),np.zeros((7,7))],[np.zeros((7,7)),-np.identity(7)]])
z = np.concatenate([yvec,-yvec])

x = np.zeros((len(omega),len(z)))
matrix = np.zeros((3,14,14))

for i in range(len(omega)):
    matrix[i,: ,:] = matrixCalculator(omega[i])
    matrix = np.array(matrix)
    _inf = np.linalg.inv(matrix[i,:,:])
    x[i,:] = np.dot(_inf,z)

solve_alpha_vect = np.vectorize(solve_alpha)
omegas = np.linspace(0.7,1.5,1000)
alphas = (solve_alpha_vect(omegas))

def questionA():
    print("____________________Question A____________________ \n")
    table = [['omega','condition number','significant digits']]
    for i in range(3):
        conditionNumber[i]=condition_number(matrix[i,:,:])
    
        sig_figs = 8 - math.ceil(math.log10(conditionNumber[i]))

        table.extend([[str(omega[i]),str(conditionNumber[i]),str(sig_figs)]])
    for i in range(4):
        print(table[i])
    

def questionB():
    print("____________________Question B____________________ \n")
    table = [['omega','_____bound_____','significant digits']]
    for i in range(3):
        bounds[i] = conditionNumber[i]*max_norm(5e-4*S)/max_norm(matrixCalculator(omega[i]))
        rel_for_error = bounds[i]
        sig_figs = -np.ceil(math.log10(rel_for_error))

        table.extend([[str(omega[i]),str(bounds[i]),str(sig_figs)]])
    for i in range(4):
        print(table[i])


def questionC():
    print("____________________Question C____________________ \n") 

    test_A = np.array([[2,1,1],[4,1,4],[-6,-5,3]])
    test_b= np.array([4,11,4])

    L, U =lu_factorize(test_A)
    y = forward_substitute(L, test_b)
    x = back_substitute(U,y)
    print("My forward_substitute result :", y)
    print("\nLibrary solution:", np.linalg.solve(L, test_b))
    print("\nMy back_substitute solution :", x)
    print("\nLibrary solution:", np.linalg.solve(U, y))

def questionD():
    print("____________________Question D____________________ \n") 
    table = [['______a(w)______','_____a(w+Dw)_____','_____a(w-Dw)_____','______Da1______','________Da2________','______||Da||______']]

    da=np.zeros([3])
    for i in range(3):
        M = matrixCalculator(omega[i])
        L, U =lu_factorize(M )
        y = forward_substitute(L, z)
        da[i] = bounds[i]*max(abs(z))*max(abs(x[i,:]))
        table.extend([[solve_alpha(omega[i]),solve_alpha(omega[i]+(5*(pow(10,-4)))),solve_alpha(omega[i]-(5*(pow(10,-4)))), solve_alpha(omega[i]+(5*(pow(10,-4))))-solve_alpha(omega[i]), solve_alpha(omega[i])-solve_alpha(omega[i]-(5*(pow(10,-4)))),da[i]]])
    for i in range(4):
        print(table[i])


def questionE():

    print("____________________Question E____________________ \n") 

    plot1 = plt.figure(1)
    plt.plot(omegas,alphas)
    plot1.suptitle('E(1)', fontsize=20)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('α(ω)', fontsize=16)
    plt.grid()
    plt.plot()
    print('There is a plot with label E(1) as an answer to this question')

def questionF():
    print("____________________Question F1____________________ \n") 

    A1 = np.array([[1,2],[3,4],[5,6]])
    b1 = np.array([1,2,3])

    Q, R1 = householder_QR_slow(A1)

    print("____________________Question F2____________________ \n") 
    VR = householder_fast(A1)
    print('VR matrix = \n',np.round(VR,3))

    print("____________________Question F3____________________ \n") 
    x, r = least_squares(A1,b1)
    print('Matrix R :\n',np.round(R1,3))
    print("\nMy least_squares result :\n", x)

def questionG():
    print("____________________Question G____________________ \n") 
    
    omegas_small = np.linspace(0.7,1.13,1000)
    alphas_small = (solve_alpha_vect(omegas_small))

    a1,r1 = polynomial_approximation(omegas_small,4)
    a2,r2 = polynomial_approximation(omegas_small,6)
    print('The polynomial coeficients for P for n=4: \na0 =',a1[0],'\na1 =',a1[1],'\na2 =',a1[2],'\na3 =',a1[3],'\na4 =',a1[4])
    print('\n There is a plot with label g(3) as an answer to the question g3')

    j = 0
    P4 = np.zeros(len(omegas_small))
    for omega in omegas_small:
        for i in range(len(a1)):
            P4[j]+=a1[i]*(omega**(2*i))
        j+=1

    plot2 = plt.figure(2)
    plot2.suptitle('Approximation with simple polyonymic ', fontsize=20)
    plt.ylabel('P(ω)', fontsize=16)
    plt.xlabel('ω', fontsize=16)
    plt.scatter(omegas_small,alphas_small)
    plt.plot(omegas_small,P4,label = 4)

    j = 0
    P6 = np.zeros(len(omegas_small))
    for omega in omegas_small:
        for i in range(len(a2)):
            P6[j]+=a2[i]*(omega**(2*i))
        j+=1

    plt.plot(omegas_small,P6,label = 6)
    plt.legend()
    print("---------------IGNORE THIS ERROR---------------------------------------------")
    plot3 = plt.figure(3)
    plt.plot(omegas_small,-np.log10((np.abs(alphas_small-P4))/alphas_small),label=4)
    plt.plot(omegas_small,-np.log10((np.abs(alphas_small-P6))/alphas_small),label=6)
    plot3.suptitle('g(3)', fontsize=20)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('Δa', fontsize=16)
    plt.legend()
    print("-----------------------------------------------------------------------------")

    sig_fig1 =-np.median(np.log10(np.divide(np.abs(P4-alphas_small),np.abs(alphas_small))))
    
    sig_fig2 =-np.median(np.log10(np.divide(np.abs(P6-alphas_small),np.abs(alphas_small))))
    print('\nThe significant digits for n=4 are : ',sig_fig1)
    print('\nThe significant digits for n=6 are : ',sig_fig2)

def questionH():
    print("____________________Question H____________________ \n") 

    a3,r3 = fractional_polynomial_approximation(omegas,2)
    a4,r4 = fractional_polynomial_approximation(omegas,4)
    print('The polynomial coeficients of Q for n =2: ')
    print('\na0 =',a3[0],'\na1 =',a3[1],'\na2 =',a3[2],'\nb1 =',a3[3],'\nb2 =',a3[4])

    Q2 = np.zeros(len(omegas))
    Q4 = np.zeros(len(omegas))
    j=0
    for omega in omegas:
        Q2[j]=(a3[0]+(a3[1]*omega)+(a3[2]*(omega**2)))/(1+(-a3[3]*omega)+(-a3[4]*(omega**2)))
        Q4[j]=(a4[0]+(a4[1]*omega)+(a4[2]*(omega**2))+(a4[3]*(omega**3))+(a4[4]*(omega**4)))/(1+(-a4[5]*omega)+(-a4[6]*(omega**2))+(-a4[7]*(omega**3))+(-a4[8]*(omega**4)))
        j+=1
        
    print("---------------IGNORE THIS ERROR---------------------------------------------")
    plot5 = plt.figure(5)
    plot5.suptitle('Approximation with fractional polyonymic ', fontsize=20)
    plt.scatter(omegas,alphas)
    plt.plot(omegas,Q2,label = 2)
    plt.plot(omegas,Q4,label = 4)
    plt.ylabel('Q(ω)', fontsize=16)
    plt.xlabel('ω', fontsize=16)
    plt.legend()

    plot7 = plt.figure(7)
    plt.plot(omegas,-np.log10((np.abs(alphas-Q2))/alphas),label=2)
    plt.plot(omegas,-np.log10((np.abs(alphas-Q4))/alphas),label=4)
    plot7.suptitle('h(2)', fontsize=20)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('Δa', fontsize=16)
    plt.legend()
    plt.plot()
    print("-----------------------------------------------------------------------------")
    print('\n There is a plot with label h(2) as an answer to the question h2')

conditionNumber, bounds =bounds_and_condition_numbers()

questionA()
questionB()
questionC()
questionD()
questionE()
questionF()
questionG()
questionH()

plt.show()
