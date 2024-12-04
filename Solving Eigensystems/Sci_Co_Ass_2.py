import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as NA
#Data
#region Data
Kmat = np.load("Chladni-Kmat.npy")
# A1-A3 should work with any implementation
A1   = np.array([[1,3],[3,1]]);
eigvals1 = [4,-2];

A2   = np.array([[3,1],[1,3]]);
eigvals2 = [4,2];

A3   = np.array([[1,2,3],[4,3.141592653589793,6],[7,8,2.718281828459045]])
eigvals3 = [12.298958390970709, -4.4805737703355,  -0.9585101385863923];

# A4-A5 require the method to be robust for singular matrices 
A4   = np.array([[1,2,3],[4,5,6],[7,8,9]]);
eigvals4 = [16.1168439698070429897759172023, -1.11684396980704298977591720233, 0]


A5   = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]);
eigvals5 = [68.6420807370024007587203237318, -3.64208073700240075872032373182, 0, 0, 0];

# A6 has eigenvalue with multiplicity and is singular
A6  = np.array(
    [[1.962138439537238,0.03219117137713706,0.083862817159563,-0.155700691654753,0.0707033370776169],
       [0.03219117137713706, 0.8407278248542023, 0.689810816078236, 0.23401692081963357, -0.6655765501236198],
       [0.0838628171595628, 0.689810816078236,   1.3024568091833602, 0.2765334214968566, 0.25051808693319155], 
       [-0.1557006916547532, 0.23401692081963357, 0.2765334214968566, 1.3505754332321778, 0.3451234157557794],
       [0.07070333707761689, -0.6655765501236198, 0.25051808693319155, 0.3451234157557794, 1.5441014931930226]]);
eigvals6 = [2,2,2,1,0]


test_matrices = [A1,A2,A3,A4,A5,A6]
test_matrices_names = ['A1','A2','A3','A4','A5','A6']
test_eigenvalues = [eigvals1,eigvals2,eigvals3,eigvals4,eigvals5,eigvals6]
#endregion
#Old code
#region old_code
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

def back_substitute(U,Y):
    X = np.zeros(len(U))
    if U[-1,-1] == 0:
        U[-1,-1] = 1
    X[-1] = Y[-1]/U[-1,-1]
    for k in range(len(U)-2,-1,-1):
        X[k] = (Y[k] - np.dot(U[k,k+1:len(U)],X[k+1:len(U)]))/U[k,k]
    return X

def problem_flipper(L,y): #Transforms the back_substitute system into forward_substitute system
    _U = np.flipud(np.fliplr(L))
    _y = np.flipud(y)
    return _U, _y

def array_to_vector(x):
    x.shape = (1, x.shape[0])
    return x

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

def maxnorm(M):
    try:
        temp = []
        for i in range(len(M)):
            temp.append(np.sum(abs(M[i,:])))
    except:
        temp = M
    return max(temp)

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
    for k in range(min(n,m) - (n==m)):
        v = R[k:,k].copy()
        v[0] +=  np.copysign(norm(v),v[0])
        u = v/norm(v) 
        R[k:,k:] += -2*np.dot(u,R[k:,k:])*u[:,NA]
        Q[k:,:] += -2*np.dot(u,Q[k:,:])*u[:,NA]
    return Q,R

def least_squares(A,b):
    Q, R = householder_QR_slow(A)
    b_=np.matmul(Q,b.copy())
    minimum = min(np.shape(R))
    b_ = b[:minimum]
    R_ = R[:minimum,:]
    r=0
    for i in range(len(A),len(b)):
        r+=b[i]**2
    return back_substitute(R_,b_)

#endregion
#Chladni Basis
#region Chladni_basis
basis_set = np.load("chladni_basis.npy")

def vector_to_function(x,basis_set):
    return np.sum(x[:,None,None]*basis_set[:,:,:],axis=0) 
    
def show_waves(x,basis_set=basis_set):
    fun = vector_to_function(x,basis_set)
    plt.matshow(fun,origin='lower',extent=[-1,1,-1,1])

def show_nodes(x,basis_set=basis_set):
    fun   = vector_to_function(x,basis_set)
    nodes = np.exp(-50*fun**2)  
    plt.matshow(nodes,origin='lower',extent=[-1,1,-1,1],cmap='PuBu')

def show_all_wavefunction_nodes(mode,centers,U,lams,basis_set=basis_set):
    idx = np.abs(lams).argsort()
    lams, U = lams[idx], U[:,idx]

    N = U.shape[0]    
    m,n = 5,3
    fig, axs = plt.subplots(m,n,figsize=(15,25))
    
    for k in range(N):
        (i,j) = (k//n, k%n)
        fun = vector_to_function(U[:,k],basis_set)
        axs[i,j].matshow(np.exp(-50*fun**2),origin='lower',extent=[-1,1,-1,1],cmap='PuBu')
        axs[i,j].set_xticklabels([])
        axs[i,j].set_yticklabels([])
        if mode == 1:
            axs[i,j].set_title(r"$center = {:.2f}$ , $\lambda = {:.2f}$".format(centers[k],lams[k])) 
        if mode == 0:
            axs[i,j].set_title(r"$\lambda = {:.2f}$".format(lams[k]))

#endregion
def  gershgorin(A):
    #this function has a matrix as an input and outputs some centers and some radii
    #The area that is described from the above include the eigenvalues
    m, n = np.shape(A)
    centers = np.diagonal(A)
    radii = np.zeros(n)
    for i in range(n):
        radii[i] =  np.abs(np.sum(A[i,:])- centers[i])
    return centers, radii

def  rayleigh_qt(A,x):
    #Here we calculate the eigenvalue where A is the matrix and x the eigenvector
    lamda = (x.T@A@x)/(x.T@x)
    return lamda

def power_iterate(A,x0):
    #here we applythe matrix a again and again to the random nonzero vector x0 until 
    #its eigenvector that we calculate is less than 0,001 close to the previous one
    k=0
    temp=10
    while np.linalg.norm(temp-x0)>10**-6:
        temp = x0
        x0 = A@x0
        x0 = x0/np.linalg.norm(x0)
        k+=1
         
    return x0, k

def inverse_power_iterate(A,x0,shift):
    A_shift = A-shift*np.eye(len(A))
    for k in range(2000):
        y_new = least_squares(A_shift,x0)

        y_new = y_new/norm(y_new)
        if y_new[0]<0:   
            y_new *= -1
        shift = rayleigh_qt(A,y_new)

        if norm(y_new - x0)<0.00001:
            break
        x0 = y_new.copy()    
    return x0, k

def rayleigh_iterate(A,x0,shift0):
    x0,k1 = inverse_power_iterate(A,x0, shift0)
    for k in range(3000):
        A_shift = A-shift0*np.eye(len(A))
        y_new = least_squares(A_shift,x0)
        y_new = y_new/norm(y_new)
        if y_new[0]<0: 
            y_new *= -1
        shift0 = rayleigh_qt(A,y_new)
        if norm(y_new - x0)<10**-10:
            break
        y = y_new.copy()      
    return y_new,k

#Execution of code
def question_a():
    print('________________________________________Question a________________________________________')

    centers, radii = gershgorin(Kmat)
    print("we have to following areas")
    limit = np.max(np.abs(centers)+radii)
    _ , ax = plt.subplots() 
    for i in range(len(centers)):
        print('center =',centers[i],' radius =',radii[i])
        circle1 = plt.Circle((centers[i], 0), radii[i], color='r')
        ax.add_patch(circle1)
    ax.set_xlim(0,2*limit)
    ax.set_ylim(-limit,limit)

def question_b():
    print('________________________________________Question b________________________________________')
    print('b3')

    for A in test_matrices:
        m,n = np.shape(A)
        x0 = np.ones(n)
        x, k= power_iterate(np.array(A),x0)
        l = rayleigh_qt(A,x)
        res = norm(A@x - l*x)
        print('A =',A)
        print('eigenvalue =',np.round(rayleigh_qt(A,x),3))
        print("eigenvector = ",x)
        print("steps = ", k)
        print("residual = ", res)
        print('--------------------')
    print('b4')
    
    m,n = np.shape(Kmat)
    x0 = np.ones(n)
    x, k = power_iterate(np.array(Kmat),x0)
    l = rayleigh_qt(Kmat,x)
    res = norm(Kmat@x - l*x)
    show_waves(x,basis_set)
    show_nodes(x,basis_set)
    print('Kmat')
    print('eigenvalue =',np.round(rayleigh_qt(Kmat,x),3))
    print("eigenvector = ",x)
    print("steps = ", k)
    print("residual = ", res)

def question_c():
    print('________________________________________Question c________________________________________')

    results = np.empty((6,4,10)).tolist()
    #in this loop I aply the code to all test matrices
    for i in range(6):
        
        print('______________',test_matrices_names[i],'________________')
        A = test_matrices[i]
        centers,radii = gershgorin(A)
        approximations = np.concatenate([centers,[np.min(centers-radii),np.max(centers+radii)]])
        eigenvalues = []
        iterations = []
        residuals = []
        j=0
        #approximations is made from all of the centers +- radious
        for approx in approximations:
            
            table = [['Eigenvalues','Iterations','Sig. Digits in residual']]
            x, k = rayleigh_iterate(A,np.random.uniform(0,1,len(A)), approx)
            iterations.append(k)
            results[i][1][j] = k
            eigenvalues.append(np.round(rayleigh_qt(A,x),4))
            results[i][2][j] = np.round(rayleigh_qt(A,x),4)
            residuals.append(norm(np.dot(A,x) - eigenvalues[-1]*x))
            results[i][3][j] = -np.floor(np.log10(norm(np.dot(A,x) - eigenvalues[-1]*x)))
            table.extend([[results[i][2][j],results[i][1][j],results[i][3][j]]])
            j+=1
        
            for k in range(2):
                print(table[k])
                
        print('____________________________________________________________________________________')

def question_d():
    print('________________________________________Question d________________________________________')
    A = Kmat 
    results = np.empty((2,50)).tolist()
    centers,radii = gershgorin(A)
    approximations = np.concatenate([centers,[np.min(centers-radii),np.max(centers+radii)]])
    eigenvalues = []
    eigenvectors = []
    j=0
    for approx in approximations:
        table = [['Eigenvalues','Eigenvectors']]
        x, k = rayleigh_iterate(A,np.random.uniform(0,1,len(A)), approx)
        eigenvectors.append(x)
        results[0][j] = np.round(x,4)
        eigenvalues.append(np.round(rayleigh_qt(A,x),4))
        results[1][j] = np.round(rayleigh_qt(A,x),4)
        table.extend([[results[1][j],results[0][j]]])
        j+=1
        
        for k in range(2):
            print(table[k])
            



question_a()
question_b()
question_c()
question_d()
plt.show()