import os
import math
#import random
import numpy as np
from scipy import linalg
from scipy import optimize
import pynauty #please figure out how to use this
import cvxpy as cvx
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#from matplotlib import animation

# GLOBAL VARIABLES
ADJACENCY_TOL = 1e-5
INITIAL_ADJACENCY_TOL = 1e-3
SAME_COORDS_TOL = 1e-5

INITIAL_STEP_SIZE = 5e-2
STEP_SIZE = 5e-3
TOL_NEWTON = 9e-16 # doesn't this seem really small?
MAX_STEP_NEWTON = 0.02
X_TOL_MAX = 20*STEP_SIZE
X_TOL_MIN = STEP_SIZE/8
ORTH_TOL = 2*STEP_SIZE

#convert adjacency dictionary (which is already the adj matrix in a hash table) to a binary vector

class Node(object):
    def __init__(self, d): #d will be indices of the clusters w/ that adj. matrix
        self.data = d
        self.left = None
        self.right = None
    def insert(self, d): #returns boolean = whether d was successfully inserted
        if self.data == d:
            return False
        elif d < self.data: #go left
            if self.left: #if left node already has smth, keep searching
                return self.left.insert(d)
            else:
                self.left = Node(d)
                return True
        else: #go right
            if self.right: #if right node already has smth, keep searching
                return self.right.insert(d)
            else:
                self.right = Node(d)
                return True
    def find(self, d):
        if self.data == d:
            return True
        elif d < self.data and self.left:
            return self.left.find(d)
        elif d > self.data and self.right:
            return self.right.find(d)
        return False
    def preorder(self, l):
        l.append(self.data)
        if self.left:
            self.left.preorder(l)
        if self.right:
            self.right.preorder(l)
        return l
    def postorder(self, l):
        if self.left:
            self.left.postorder(l)
        if self.right:
            self.right.postorder(l)
        l.append(self.data)
        return l
    def inorder(self, l): #**will give adj. vectors in ascending order
        if self.left:
            self.left.inorder(l)
        l.append(str(self.data))
        if self.right:
            self.right.inorder(l)
        return l

class bst(object):
    def __init__(self):
        self.root = None
    def insert(self, d): #return T if successfully inserted, F if already exists
        if self.root: #tree already has stuff
            return self.root.insert(d)
        else:
            self.root = Node(d)
            return True
    def find(self, d): #return T if d is found in tree
        if self.root:
            return self.root.find(d)
        else:
            return False
    def remove(self, d):
        pass #complicated & don't need for now
    #returning ordered list of data elements in bst
    def preorder(self):
        if self.root:
            return self.root.preorder([])
        else:
            return []
    def postorder(self):
        if self.root:
            return self.root.postorder([])
        else:
            return []
    def inorder(self):
        if self.root:
            return self.root.inorder([])
        else:
            return []   

class cluster(object):
    def __init__(self, vector):
        self.adj_vector = vector
    def __lt__(self, other): #if self < other
        #keep comparing indices until 1 is less
        for i in range(len(self.adj_vector)):
            if self.adj_vector[i] < other.adj_vector[i]:
                return True
            elif self.adj_vector[i] > other.adj_vector[i]:
                return False
        return False
    def __gt__(self, other): #if self > other
        #keep comparing indices until 1 is less
        for i in range(len(self.adj_vector)):
            if self.adj_vector[i] > other.adj_vector[i]:
                return True
            elif self.adj_vector[i] < other.adj_vector[i]:
                return False
        return False 
    def __str__(self): #np string is not actually a good representation
        #... so try to change it to actually store the array itself
        return np.array2string(self.adj_vector)

#def add_cluster():

# METHODS FOR RIGIDITY AND DIMENSION

def adj_dict_to_vector(graph): #graph is Graph object
    vtxs = graph.number_of_vertices
    adj_matrix = np.zeros((vtxs, vtxs))
    adj_dict = graph.adjacency_dict
    for vtx in adj_dict: #iterate over the keys
        for contact in adj_dict[vtx]: #symmetric matrix
            adj_matrix[vtx][contact] = 1
            adj_matrix[contact][vtx] = 1
    #print(adj_matrix)
    return adj_matrix
    #return adj_matrix.flatten()

def adj_matrix_to_dict(A):
    adj_dict = {}
    for i in range(len(A)):
        #print("i =",i)
        for j in range(i+1, len(A[0])): 
            #print("\nj =",j)
            #i+1 to skip the diagonal (sphere can't contact itself so diagonal always 0)
            if A[i][j] == 1:
                if i in adj_dict:
                    adj_dict[i].append(j)
                else:
                    adj_dict[i] = [j]
    return adj_dict

def adj_matrix(x,d,n, A = [], returnA = True):
    '''
    Return n x n adjacency matrix of cluster as a list of lists (not numpy array).
    '''
    # CREATE ADJACENCY MATRIX FROM VECTOR X
    #print("len(A) =", len(A))
    if returnA: #if you want to return A, make the adjacency matrix
        #print("x =", x)
        A = []
        for i in range(n):
            new_row = []
            for j in range(n):
                norm = 0
                for k in range(d):
                    norm += (x[d*i+k] - x[d*j+k])**2
                #print("norm =", norm)
                if i!=j and abs(norm - 1) <= ADJACENCY_TOL: #otherwise it will think each sphere contacts itself
                    new_row.append(1)
                else:
                    new_row.append(0)
            #print("new row =", new_row)
            A.append(new_row)
    return A

def system_eqs(x, d, n, A, tol = 1e-6):
    '''
    Given cluster coordinates x (vector sized nd) and its adjacency matrix, return a vector with elements
    corresponding to all contacts (where A[i][j]=1) and whether the current coordinates (since this is used
    when x has been perturbed) still fulfill the original equation. Element is 0 if a contact still holds; 
    otherwise, it is the distance between them. 
    Length is the number of contacts (e.g. 3n-6) + d(d+1)/2 for the constrained vertices.
    *** Should an error be raised if the distance is negative? (spheres can't intersect)
    '''
    distances = []
    #overlaps = 0
    for i in range(len(A)):
        #print("i =",i)
        for j in range(i+1, len(A[0])): 
            #print("\nj =",j)
            #i+1 to skip the diagonal (sphere can't contact itself so diagonal always 0)
            if A[i][j] == 1:
                #i is always less than j b/c we only iterate through upper triangle
                norm = 0 # is actually the square of the norm (of x_i - x_j)
                for k in range(d):
                    norm += (x[d*i+k] - x[d*j+k])**2
                #print("norm =", norm)
                norm = math.sqrt(norm)
                # if norm - 1 < -1e-14:
                    # print('overlap sphere', i, '&', j, ':', norm-1)
                    # overlaps += 1
                if -1e-13 < norm - 1 < tol: 
                    distances.append(0) # contact is still present
                else:
                    distances.append(norm - 1)
    distances += [0 for constraint in range(int(d*(d+1)/2))]
    #print(overlaps, "overlaps")
    #print("system_eqs() returns", distances)
    return np.array(distances)
    #return np.array(distances).T

def newton_system(x0, x, d, n, A, v_j):
    distances = system_eqs(x, d, n, A)
    orth_test = 100*(np.inner(np.array(x) - np.array(x0), v_j)) ** 2
    return np.append(distances,orth_test)

def rigidity_matrix(x, d, n, A = [], returnA = True):
    # can iterate through adjacency matrix A and look for 1s (indicate contacts)
    #kd tree
    #tolerance for adjacency = 10^-3 or 10^-5
    
    # CREATE ADJACENCY MATRIX FROM VECTOR X
    A = adj_matrix(x, d, n, A, returnA)

    R = []
    for i in range(len(A)):
        #print("i =",i)
        for j in range(i+1, len(A[0])): 
            #print("\nj =",j)
            #i+1 to skip the diagonal (sphere can't contact itself so diagonal always 0)
            if A[i][j] == 1:
                #i is always less than j b/c we only iterate through upper triangle
                new_row = []
                prezeros = d*i
                midzeros = d*(j-i-1)
                postzeros = d*(len(A[0])-j-1)
                iminj = [] #p_ik - p_jk
                jmini = [] #-(p_ik-p_jk)
                
                #print("d*i =",d*i,"\t(d+1)*i =",d*i+d)
                #print("d*j =",d*j,"\t(d+1)*j =",d*j+d)
                
                for coord_i, coord_j in zip(x[d*i:d*i+d], x[d*j:d*j+d]):
                    iminj.append(coord_i - coord_j)
                    jmini.append(coord_j - coord_i)

                #print("iminj =", iminj)
                #print("jmini =", jmini)

                new_row += [0 for zero in range(prezeros)]
                new_row += iminj
                new_row += [0 for zero in range(midzeros)]
                new_row += jmini
                new_row += [0 for zero in range(postzeros)]
                
                #print("new row is", len(new_row), "elements long")
                assert len(new_row) == n*d

                R.append(new_row)
    
    R = constrain(R, d, n)
    #print(R)
    #print("Returning A?:", returnA)
    if returnA:
        return (R, A) #rigidity and adjacency
    else:
        return R

def constrain(matrix, d, n):
    '''
    Adds rows of constrained vertices to the end of the rigidity matrix, a 2d numpy array. 
    Should mutate input matrix. (helper function)
    Returns constrained matrix.
    '''
    matrix = np.array(matrix)

    s_j = [] #indices/vertices to constrain
    ind = 0
    for sphere in range(d):
        ind += sphere
        s_j += list(range(ind, d*(sphere+1)))
        ind = d*(sphere+1)
    
    #print("constrained vertices:", s_j) #should be 1,2,3,4,5,7,8,9,10,13,14,15,19,20,25

    for coord in s_j:
        new_row = [] #will be added to rigidity matrix
        prezeros = coord
        postzeros = n*d - coord - 1
        new_row += [0 for zero in range(prezeros)]
        new_row.append(1) #for e_s_j^T
        new_row += [0 for zero in range(postzeros)]
        #print("new row to add", new_row)
        #assert len(new_row) == n*d
        matrix = np.append(matrix, [new_row], axis=0)
    #print("matrix to return:", matrix)
    return matrix

def sign_def(matrix): #returns bool
    '''
    Returns True if input matrix is positive or negative sign definite; False otherwise.
    '''
    eigs = linalg.eigvals(matrix)
    #what to do if the eigenvalues are complex??
    if eigs[0] > 0: #test for positive
        for eig in eigs:
            if eig <= 0:
                return (False, None)
    elif eigs[0] < 0: #test negative sign definite
        for eig in eigs:
            if eig >= 0:
                return (False, None)
    else: #first term 0
        return (False, None)
    return (True, np.sign(eigs[0]) * min([abs(eig) for eig in eigs])) # returns eig w/ min abs value

def is_rigid(RA, d, n): #R is rigidity matrix (should be 2d numpy array)
    '''
    Returns 1 if cluster if 1st order rigid, 2 if it is pre-stress stable.
    Returns 0 (maybe) if the cluster is not rigid.
    '''
    R, A = RA

    #TEST FIRST ORDER RIGIDITY
    # if right null space V, dim = n_v = 0 --> return 1 (for 1st order rigid)
    #print("dimension of R:", R.shape[0], "x", R.shape[1])
    right_null = linalg.null_space(R) #V, gives orthonormal basis
    '''
    instead of null_space(), check min singular value, if nonzero, kernel is empty
    '''
    n_v = right_null.shape[1]
    #print("dim(right null space) =", n_v)
    
    if n_v == 0: #(N,K) array where K = dimension of effective null space
        print("First-order rigid")
        return (1,right_null)
    
    transpose = R.T
    #print("TRANSPOSE:", transpose)
    left_null = linalg.null_space(transpose)#W
    print("SHAPE:", left_null.shape)
    n_w = left_null.shape[1]
    #print("dim(left null space) =", n_w)
    if n_w == 0: # this seems to mean the cluster is hypostatic
        print("Not rigid")
        print("Dimension =", n_v) # for hypostatic clusters, D = n_v = 1
        return (0,right_null) #??? return n_v
    else: 
        #TEST PRE-STRESS STABILITY
        '''
        for m in range(n_w): #iterate from 1 to n_w
            b = [0 for zero in range(n_v)] #b has size n_v
            b[m] = 1 #b_m = e_m
            #now make Q and do b*Q
            #print("b =", b)
            Q_m = [] #dimension n_v x n_v
            for i in range(n_v):
                new_row = []
                for j in range(n_v):
                    #print("left_null.T[m] =", left_null.T[m])
                    #print("dimension of left_null.T[m]:", left_null.T[m].shape)
                    Rv = rigidity_matrix(right_null.T[i], d, n, A, False)
                    #print("dimension of Rv:", Rv.shape)
                    wR = np.matmul(left_null.T[m], Rv)
                    # does it need to be the transpose of right_null[i]?????????
                    new_row.append(np.matmul(wR, right_null.T[j]))
                Q_m.append(new_row)
            print(Q_m)
            if sign_def(Q_m)[0]:
                print("Pre-stress stable")
                return (2,right_null)
            '''
        if sdp(right_null, left_null, A, d) is not None:
            print("Pre-stress stable")
            return (2, right_null)
    print("Unable to determine rigidity")
    # numerical_dimension(x, d, n, A, right_null)
    return (-1,right_null) #????

def omega(w, A, d):
    '''
    Returns stress matrix Omega with dimensions dn x dn.
    Omega acts on two flexes (u,v in right kernel) as u.T*Omega*v = sum( w_ij (u_i - u_j)(v_i - v_j)) - in R
    "Large" omega is the Kronecker product of the n x n stress matrix with d x d identity.
    '''
    # contacts btwn i, j should be ordered so that smallest i come first

    # make "small" n x n stress matrix first
    n = len(A)
    s = np.zeros((n,n)) #small stress matrix
    w_index = 0
    for i in range(n):
        #print("i =",i)
        for j in range(i+1, n): 
            #print("\nj =",j)
            #i+1 to skip the diagonal (sphere can't contact itself so diagonal always 0)
            if A[i][j] == 1:
                s[i][j] = -w[w_index]
                s[j][i] = -w[w_index]
                w_index += 1
    # set diagonal entries
    for i in range(n): # len(A) = n
        rowsum = 0
        for j in range(n):
            if j != i:
                rowsum += s[i][j]
        s[i][i] = -1*rowsum
        #assert sum([x for x in s[i]]) <= 1e-5 # rowsum must be 0
    # for i in range(n):
    #     assert sum([s[i][j] for j in range(n)]) == 0 # colsum must be 0
    #print("\nSMALL STRESS MATRIX:\n", s)
    I = np.eye(d)
    stress = np.kron(s, I)
    return stress

def sdp(V, W, A, d):
    '''
    Input: V (np array) = basis of right null space; W (np array) = basis of left null space.
    Returns tuple (stress_opt, opt_eig) representing the optimal stress w in R^m and corresponding eigenvalue. (?)
    Raises ValueError if the test was unsuccessful.
    '''
    # print("shape of W before:", W.shape)
    
    # print("shape of W after:", W.shape)
    # M_i = V.T * Omega(w_i) * V
    n_v = V.shape[1]
    n_w = W.shape[1]
    I = np.identity(n_v)
    W = W.T
    print("n_w =", n_w)

    # handle special cases: either n_w = 1 or n_v = 1
    if n_w == 1:
        # only 1 self-stress
        M = np.matmul(V.T, omega(W[0], A, d))
        M = np.matmul(M, V)
        M = (M + M.T)/2
        s_def, min_eig = sign_def(M) # bool, num
        if s_def:
            return (np.sign(min_eig)*W[0], abs(min_eig))
        else:
            #raise ValueError("test unsuccessful: M not sign definite for n_w = 1")
            return None
    elif n_v == 1:
        # only 1 infinitesimal flex, each M is a scalar
        Ms = [] # store the Ms, find max abs value
        maxM_i = None
        sign_max = 1
        for i in range(n_w):
            M = np.matmul(V.T, omega(W[i], A, d)) # V[:,None] to take transpose of 1d vector (??)
            M = np.matmul(M, V)
            M = float(M) # M should be a scalar
            if M != 0: # should I watch out for numerical tolerance?
                Ms.append(abs(M)) # only store positive values in Ms
                if maxM_i is None or abs(M) > abs(Ms[maxM_i]):
                    maxM_i = i
                    sign_max = np.sign(M)
    
        if len(Ms) == 0:
            #raise ValueError("test unsuccessful: all M = 0 for n_v = 1")
            return None
        else:
            return (sign_max * W[maxM_i], Ms[maxM_i])

    t = cvx.Variable() # scalar
    a = cvx.Variable(n_w)
    #X = cvx.Variable((n_v,n_v),symmetric=True)
    X = np.zeros((n_v,n_v))
    
    for i in range(n_w):
        M = np.matmul(V.T, omega(W[i], A, d))
        M = np.matmul(M, V)
        M = (M + M.T)/2 # symmetrize to remove imaginary eigenvalues
        X += a[i]*M - t*I
    constraints = [X >> 0, cvx.norm(a)<=1]

    prob = cvx.Problem(cvx.Maximize(t),constraints)
    prob.solve()
    t_opt = prob.value
    a_opt = a.value
    if t_opt <= 0:
        #raise ValueError("test unsuccessful, t_opt is nonpositive")
        print("test unsuccessful, t_opt is nonpositive")
        return None
    print("STATUS:", prob.status)
    if prob.status == cvx.OPTIMAL:
        print("\na_opt =", a_opt)
        print("\nt_opt =", t_opt)
        return 0 # only checking if it's not None
        stress_opt = np.zeros(n_w)
        for i in range(n_w):
            stress_opt += np.array(a_opt[i]*W[i]) # this is a cvx variable - np might not work here?
        return (stress_opt, t_opt) # t_opt is the optimal eigenvalue
    else:
        raise ValueError("CVX did not find optimal solution")

def numerical_dim(x, d, n, A, right_null): # if is_rigid returned 0 or -1
    '''
    x is the coordinates of the cluster (the "approximate solution").
    right_null is the basis of the right null space of R(x), which was determined and passed in via
    is_rigid(). right_null must be nonempty (or this function would not have been called).
    Return estimated dimension of cluster.
    '''
    print("\nfinding numerical dim...")
    print("starting x:", x)
    #failed_newtons = []
    basis = np.empty((0,d*n)) # what are the dims here?
    for v_j in right_null.T: # extracts the "vertical" basis vectors
        #print("v_j =", v_j)
        for sign in ['+','-']: # take step in both directions
            if sign == '+':
                x_step = [coord + INITIAL_STEP_SIZE*v for coord, v in zip(x, v_j)]
            else:
                x_step = [coord - INITIAL_STEP_SIZE*v for coord, v in zip(x, v_j)]

            # project onto constraints using Newton's method 
            projx, iters = newtons(newton_system, rigidity_matrix, x_step, d, n, A, v_j, graph=True) 
            #graph(x,projx,'Projection in num dim method',np.linalg.norm(projx - x))
            #print("\nanalytic method returns", is_rigid())
            if iters == -1: # exceeded maximum iterations
                return (-1, []) 
                raise RuntimeError("Newton's method did not converge")
            # if abs(np.inner(projx - x_step, v_j)) > ORTH_TOL: # what tol value to use??
            #     raise RuntimeError("Newton result not orthogonal; inner product =", np.inner(projx - x_step, v_j))
                #failed_newtons.append((v_j, 'newton result was not orth for ' + sign))

            # assuming x, projx, are both np vectors/arrays:
            tanv_plus = projx - x
            norm_plus = np.linalg.norm(tanv_plus)
            #print("distance from original =", norm_plus)
            #print("tanv_plus = projx - x =", tanv_plus)
            if (norm_plus > 10*X_TOL_MAX or norm_plus < X_TOL_MIN):
                if norm_plus > X_TOL_MAX:
                    print("too big:", norm_plus)
                else:
                    print("too small")
                print("rejecting vector")
                pass # reject vector
            else:
                # project onto current estimate of B; projection matrix P = A(A.T A)^-1 A.T
                if len(basis) == 0:
                    print("size of added:", (tanv_plus/np.linalg.norm(tanv_plus)).shape)
                    #basis.append(tanv_plus/np.linalg.norm(tanv_plus))
                    basis = np.append(basis, [tanv_plus/np.linalg.norm(tanv_plus)], axis = 0)
                else:
                    proj = projection(tanv_plus,basis).T # this probably needs to be basis.T
                    orthonorm = tanv_plus - proj # orthogonal portion to the projection - check this
                    if np.linalg.norm(orthonorm) > ORTH_TOL:
                        print("size:", (orthonorm/np.linalg.norm(orthonorm)).shape)
                        #basis.append(orthonorm/np.linalg.norm(orthonorm))
                        basis = np.append(basis, orthonorm/np.linalg.norm(orthonorm), axis = 0)
    #print("basis =", basis)
    print("exit numerical_dim()\n")
    return (len(basis), basis)

def projection(v, A):
    '''
    Projects vector v onto the column space of A (using the SVD of A).
    '''
    #print("A =", A)
    A = np.array(A)
    #print("dimensions of A:", A.shape)
    v = v[:,None]
    #print("dimensions of v:", v.shape)
    try:
        x, res, rank, s, = np.linalg.lstsq(A,v,rcond=None)
        project = np.matmul(A, x)
    except:
        x, res, rank, s, = np.linalg.lstsq(A.T,v,rcond=None)
        project = np.matmul(A.T, x)
    #project = np.matmul(A, x)
    return project

def newtons(F, J, x, d, n, A, v_j, eps=TOL_NEWTON, graph = False): 
    """
    Solve nonlinear system F=0 by Newton's method. J(x) is the Jacobian of F(x). 
    At input, x holds the start value. The iteration continues until ||F|| < eps.
    Step size per iteration limited by MAX_STEP_NEWTON.
    """
    print("\nBegin Newton's:")
    x0 = x[:] # initial value, for use in calculating orthogonality 
    F_value = F(x0, x, d, n, A, v_j) # F is newton_system()
    inn_prod = None
    dist_from_orig = [0]
    F_norm = np.linalg.norm(F_value[:-1], ord=2)  # l2 norm of vector
    iteration_counter = 0
    print("\nin newton's, F_norm =", abs(F_norm))
    while iteration_counter < 200 and (abs(F_norm) > eps or (iteration_counter < 50 and (inn_prod is None or abs(inn_prod) > 0))): 
        # test for np.inner(projx - x_step, v_j) > ORTH_TOL
        if (iteration_counter >= 50 or (inn_prod is None or inn_prod == 0.0)): # iteration_counter >= 95
            jac = J(x, d, n, A, False)
            F_value = F_value[:-1]
        else:
            direction = [100*2*v_j[i]*inn_prod for i in range(len(x))]
            jac = np.append(J(x, d, n, A, False), [direction], axis = 0)
        print("size of J:", jac.shape)
        if jac.shape[0] == jac.shape[1]: #square matrix
            delta = np.linalg.solve(jac, -F_value)
        else:
            inv = np.linalg.pinv(jac)
            delta = np.matmul(inv,-F_value)
        # constrain maximum step size 
        delt_norm = np.linalg.norm(delta)
        print("delt norm =", delt_norm)
        if delt_norm > MAX_STEP_NEWTON:
            delta = delta*(MAX_STEP_NEWTON/delt_norm) # scale to max step size
        #print("delta =", delta)
        # on first step, project delta to be inner(delta - inner(delta,v_j) * v_j), v_j)
        # if iteration_counter <= 95:
        #     delta = np.inner(delta - np.inner(delta, v_j) * v_j, v_j)
        x = x + delta
        #print("updated x =", x)
        F_value = F(x0, x, d, n, A, v_j)
        inn_prod = F_value[-1]
        print("inner product =", inn_prod)
        F_norm = np.linalg.norm(F_value[:-1], ord=2)
        print("iteration", iteration_counter, "/ F_norm = ", F_norm)
        iteration_counter += 1
        dist_from_orig.append(np.linalg.norm(x-x0))
    if False and graph:
        plt.plot(dist_from_orig)
        plt.show()

    print("exit newton's w/ F_norm =", F_norm, "& inner product =", inn_prod, "after", iteration_counter, "iterations\n")
    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    print("length of x:", len(x))
    return x, iteration_counter

# MOVING ALONG ONE-DIMENSIONAL MANIFOLD

def combos(contacts,n):
    '''
    Return all combinations of numbers in range [0, size) of size n.
    Need to pass in contacts as list(range(size)).
    '''
    if n == 0:
        return [[]]
    lis = []
    #contacts = [i for i in range(size)]
    for i in range(len(contacts)):
        stud = contacts[i]
        rest = contacts[i+1:]
        for subcombo in combos(rest, n-1):
            lis.append([stud] + subcombo)
    return lis

def within_tol(x,d,n,len_broken,tol=INITIAL_ADJACENCY_TOL):
    '''
    Return bool based on whether 2 spheres initially not in contact are within some tolerance 1 + tolA0
    len_broken (int) is the length of the broken (1d manifold) cluster we're working with.
    broken is a list of indices that correspond to broken contacts (should only be 1 at a time).
    '''
    updated_A = adj_matrix(x, d, n)
    print(np.array(updated_A))
    contacts = system_eqs(x,d,n,updated_A, tol)
    print("# of contacts in within_tol:", len(contacts))
    if len(contacts) > len_broken: # new contact found between spheres
        return updated_A # does this make sense to return?
    return None #no new contacts (default)

def update_A(A, breaks):
    '''
    Change 1s to 0s in the A matrix at all indices in breaks (a list, e.g. [0,3,7]). 
    Shouldn't mutate input.
    '''
    newA = np.copy(A)
    index = 0 # index counter
    for i in range(len(newA)):
        for j in range(i+1, len(newA[0])): 
            if newA[i][j] == 1 and index in breaks:
                newA[i][j] = 0 # contact removed
                newA[j][i] = 0
            index += 1
    return newA

def manifold(x0, d, n, A, breaks, basis):
    '''
    Method called only if we have a 1D solution set.
    Input: basis is np matrix B, calculated in numerical dimension method;
    A is the (initial) adjacency matrix;
    breaks (list) is a list of the indices of the contact(s) that were broken to create the manifold.
    '''
    print("\nSTART OF MANIFOLD()\nbasis =", basis)
    x = x0[:] # make a copy since we're mutating x
    dirs = [] # directions for which contacts increase - may change to set
    contacts = system_eqs(x,d,n,A) # initially, should all = 0
    len_broken = len(contacts)
    print("size of orig contacts =", len_broken)

    #for vec in basis.T:
    for vec in basis:
        new_contacts_plus = system_eqs(x + 2*INITIAL_STEP_SIZE*vec,d,n,A) # some should be != 0
        new_contacts_minus = system_eqs(x - 2*INITIAL_STEP_SIZE*vec,d,n,A)
        # print("size of new contacts =", len(new_contacts_minus))
        for broken_i in breaks:
            print("broken_i =", broken_i)
            #print(new_contacts_minus[broken_i])
            if new_contacts_plus[broken_i] > 0: # at least 1 broken contact increased (should it be all?)
                print("value =", new_contacts_plus[broken_i])
                print("\ncontact (+) increased in length!!")
                dirs.append(vec)
                break
            if new_contacts_minus[broken_i] > 0:
                print("value =", new_contacts_minus[broken_i])
                print("\ncontact (-) increased in length!!")
                dirs.append(vec)
                break

    if len(dirs) == 0:
        print("\nCouldn't find any dirs to increase length")
        return

    # for direc in dirs[:]:
    #     dirs.append(-1*direc) # this seems repetitive: we already test both directions when forming dirs

    print(len(dirs), 'directions')
    print("len_broken =", len_broken)
    #A = adj_matrix(x, d, n)
    for v_k in dirs:
        v = v_k # update this as you go
        x = x0[:]
        prev_step = x # used for when we backtrack 1 step at the end
        step_count = 0
        print("\nIterating through dirs:")
        while (step_count == 0 or within_tol(x,d,n,len_broken) is None) and step_count < 2000: # stop when you get 2 spheres close enough
            step_count += 1
            #print("step", step_count, "& v =", v)
            print("\nstep", step_count)
            # take step in tangent direction, along manifold
            step = [coord + 2*INITIAL_STEP_SIZE*v for coord, v in zip(x, v)] # is it vector x we're adding to?
            #print('\nNEW POSITION:\n', step)
            # project back onto manifold
            step, iters = newtons(newton_system, rigidity_matrix, step, d, n, A, v)
            if iters == -1: #newton's failed
                raise KeyboardInterrupt
            #newR = rigidity_matrix(step, d, n, A, False)
            newR, A = rigidity_matrix(step, d, n)
            right_null = linalg.null_space(newR) # if right_null = [] the dimension decreased: should break
            # print("right null:", right_null)
            # print("v:", v)
            v = projection(v, right_null).T[0] # check syntax - outputs something like [[]]
            prev_step = x # save prev position
            x = step # update current position

            #check dimension after step 1
            used_numdim = False
            if step_count == 1:
                print("checking dimension after step 1:")
                dim = is_rigid((newR, A), d, n)
                if dim[0] == 0 or dim[0] == -1: # analytic method didn't work
                    print("analytic didn't work, dim =", dim[0])
                    used_numdim = True
                    dim = numerical_dim(x, d, n, A, dim[1])[0]
                else:
                    dim = dim[0] # only keep the number
                    if dim == 1 or dim == 2:
                        dim = 0
                    # else:
                    #     dim = 1 # ????
                # check if dim has increased or decreased
                if dim > 1:
                    print("dimension increased")
                    return -1 # stop moving, dim increased
                elif dim < 1:
                    print("breaking, current x =", x)
                    print("used numdim:", used_numdim)
                    print("size of R:", newR.shape)
                    print("A =", np.array(A))
                    graph(x0,x,'Dimension decreased :(',np.linalg.norm(x-x0))
                    return # does this mean we moved back to the original??
                #raise RuntimeError
        #print("distance between =", np.linalg.norm(x-x0))
        graph(x0,x,'Manifold method, larger steps',np.linalg.norm(x-x0))

        x = prev_step # back up 1 step
        
        # repeat continuation with smaller step size; stop when you get 2 spheres close enough, again
        final_steps = 0
        print("\ndoing final round after", step_count, "steps")
        while (final_steps == 0 or within_tol(x,d,n,len_broken) is None) and final_steps < 200:
            print("final step #", final_steps)
            final_steps += 1
            # take step in tangent direction, along manifold
            step = np.array(x) + STEP_SIZE*v
            #step = [coord + STEP_SIZE*v for coord, v in zip(x, v.T)] # getting ragged nested arrays here
            #print("step =", step)

            # project back onto manifold
            step, iters = newtons(newton_system, rigidity_matrix, step, d, n, A, v)
            if iters == -1: #newton's failed
                raise ValueError("projection (newton's) failed")
            newR, A = rigidity_matrix(step, d, n)
            right_null = linalg.null_space(newR)
            v = projection(v, right_null).T[0]
            x = step # update current position
        graph(x0,x,'Manifold method, smaller steps',np.linalg.norm(x-x0))

        print("\nGot 2 spheres close enough")
        # Then, we project onto this new set of constraints and check if the cluster is rigid, 
        # using a new tolerance tolA to determine whether two spheres are adjacent.
        proj_final, iters = newtons(newton_system, rigidity_matrix, x, d, n, A, v)
        if iters == -1: # projection failed
            '''
            delete subsets of the new constraints until the projection succeeds
            '''
            raise NotImplementedError("delete subsets of the new constraints")
        finalR = rigidity_matrix(proj_final,d,n,A)
        final_rigidity = is_rigid(finalR,d,n)[0]
        print("final rigidity =", final_rigidity)
        # implement numerical method here?
        if final_rigidity == 1 or final_rigidity == 2: # new cluster found!
            # change implementation later
            return proj_final

def graph(x1, x2, title='', dist = 'n/a'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1s, y1s, z1s, x2s, y2s, z2s = [], [], [], [], [], []
    for i in range(0,len(x1),3):
        x1s.append(x1[i])
        y1s.append(x1[i+1])
        z1s.append(x1[i+2])
        x2s.append(x2[i])
        y2s.append(x2[i+1])
        z2s.append(x2[i+2])
    print("\n\nx1 =",x1s[1])
    print("y1 =",y1s[1])
    print("z1 =",z1s[1])
    ax.scatter(x1s, y1s, z1s, c='red')
    ax.scatter(x2s, y2s, z2s, c='blue')
    ax.set_title(title + ', distance moved = ' + str(dist))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# TEST FUNCTIONS

def parse_coords(n): #returns list of lists (of coordinates)
    #f (file) is one long string \n for new clusters
    filename = 'test-coords/n' + str(n) + '.txt'
    #filename = os.path.join('test-coords/n', str(n), '.txt')
    f = open(filename, 'r')
    clusters = []
    for line in f: #each line represents one cluster
        this_cluster = []
        for coord in line.split(): #these are all strings
            this_cluster.append(float(coord))
        clusters.append(this_cluster)
    f.close()
    return clusters

def test_hypercube():
    print("\n3D cube")
    cube3 = [0.0,0.0,0.0, 0.0,0.0,1.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 1.0,0.0,1.0, 1.0,1.0,1.0, 1.0,0.0,0.0, 1.0,1.0,0.0]
    R_cube3 = rigidity_matrix(cube3, 3, 8)

    # adj_cube3 = adj_matrix(cube3, 3, 8)
    # print("\nADJACENCY MATRIX:")
    # print(np.array(adj_cube3))
    # contacts3 = system_eqs(cube3, 3, 8, adj_cube3)
    # print("\nCONTACTS:")
    # print(contacts3)

    #print(R_cube)
    rigid_cube3 = is_rigid(R_cube3,3,8)
    print(rigid_cube3)

    print("\n4D cube") # about 1 second delay
    cube4 = []
    binary = [0.0,1.0]
    cube4 += [[a,b,c,d] for a in binary for b in binary for c in binary for d in binary]
    cube4 = np.array(cube4).flatten()
    R_cube4 = rigidity_matrix(cube4,4,16)
    rigid_cube4 = is_rigid(R_cube4,4,16)
    print(rigid_cube4)

    # print("\n5D cube") # about 15-20 seconds delay? very slow
    # cube5 = []
    # cube5 += [[a,b,c,d,e] for a in binary for b in binary for c in binary for d in binary for e in binary]
    # cube5 = np.array(cube5).flatten()
    # R_cube5 = rigidity_matrix(cube5,5,32)
    # rigid_cube5 = is_rigid(R_cube5,5,32)
    # print(rigid_cube5)

def test_simplex():
    print("\n3D Simplex (i.e. tetrahedron):\n")
    simplex3 = [0.5,0.0,-1/(2*math.sqrt(2)), \
                -0.5,0.0,-1/(2*math.sqrt(2)), \
                0.0,0.5,1/(2*math.sqrt(2)), \
                0.0,-0.5,1/(2*math.sqrt(2))]
    adj_sim3 = adj_matrix(simplex3, 3, 4)
    print("\nADJACENCY MATRIX:")
    print(np.array(adj_sim3))
    contacts3 = system_eqs(simplex3, 3, 4, adj_sim3)
    print("\nCONTACTS:")
    print(contacts3)
    rigid_sim3 = is_rigid(rigidity_matrix(simplex3,3,4),3,4)
    print(rigid_sim3)

    print("\n4D Simplex:\n")
    phi = (1 + 5**0.5)/2
    edge = 2*math.sqrt(2)
    simplex4 = [2/edge,0,0,0, 0,2/edge,0,0, 0,0,2/edge,0, 0,0,0,2/edge, phi/edge, phi/edge, phi/edge, phi/edge]
    # simplex4 = [1/(2*math.sqrt(10)), 1/(2*math.sqrt(6)), 1/(2*math.sqrt(3)), 0.5, \
    #             1/(2*math.sqrt(10)), 1/(2*math.sqrt(6)), 1/(2*math.sqrt(3)), -0.5, \
    #             1/(2*math.sqrt(10)), 1/(2*math.sqrt(6)), -1/(math.sqrt(3)), 0.0, \
    #             1/(2*math.sqrt(10)), -1*math.sqrt(3/2)/2, 0.0, 0.0, \
    #             -1*math.sqrt(2/5), 0.0, 0.0, 0.0]
    # adj_sim4 = adj_matrix(simplex4, 4, 5)
    # print("\nADJACENCY MATRIX:")
    # print(np.array(adj_sim4))
    # contacts4 = system_eqs(simplex4, 4, 5, adj_sim4)
    # print("\nCONTACTS:")
    # print(contacts4)
    rigid_sim4 = is_rigid(rigidity_matrix(simplex4,4,5),4,5)
    print(rigid_sim4)

def test_hc_rigid_clusters(start_n, end_n):
    d = 3
    first_rigid = [] #first order rigid (1)
    pre_stress = [] #pre stress stable (2)
    not_rigid = [] #(0)
    idk = [] #can't be determined (-1)]
    hypostatic = [] # contacts < 3n-6
    hyperstatic = []
    isostatic = []
    #hypo_stress = [] #hypostatic & pre-stress (should be all of them?)
    cond_1 = [] # condition numbers of first-order matrices
    cond_pre = [] # cond of pre-stress stable
    cond_w0 = [] # cond of matrices where |W| = 0
    #hypo_rigid = -2
    for n in range(start_n, end_n):
        print("\nTesting n =", n)
        clusters = parse_coords(n)
        for cluster in clusters: #cluster is x
            assert len(cluster) == d*n
            #print("cluster:", cluster)
            R = rigidity_matrix(cluster, d, n)
            contacts = 0
            
            for i in range(len(R[1])):
                for j in range(i+1, len(R[1][0])):
                    if R[1][i][j] == 1:
                        contacts += 1
            
            #print(R)
            rigid = is_rigid(R,d,n)[0] #0, 1, or 2
            if rigid == 1:
                first_rigid.append(cluster)
                cond_1.append(np.linalg.cond(R[0]))
            elif rigid == 2:
                #print("PSS CLUSTER:", cluster)
                pre_stress.append(cluster)
                cond_pre.append(np.linalg.cond(R[0]))
                # test svd
                u, s, vh = np.linalg.svd(R[0], full_matrices=True)
                min_sing = min(s)
                print("minimum singular value =", min_sing)
                if min_sing > 1e-16: # 10^-16
                    print("Kernel is empty based on singular values\n")
                # sig = np.diag(s[:-1])
                # print("null space of Sigma:", linalg.null_space(sig))
                #print("singular values:", s)
            elif rigid == 0:
                not_rigid.append(cluster)
                cond_w0.append(np.linalg.cond(R[0]))
            elif rigid == -1:
                idk.append(cluster)
            if contacts < d*n - 6:
                hypostatic.append(cluster)
                #hypo_rigid = rigid
            elif contacts > d*n - 6:
                hyperstatic.append(cluster)
            else:
                isostatic.append(cluster)
            #assert rigid != 0 # these all should be rigid
            #print("\n")

    print("For all clusters n =", start_n, "through", end_n-1)
    print("# of first-order rigid clusters:", len(first_rigid))
    print("# of pre-stress stable clusters:", len(pre_stress))
    print("# of non-rigid clusters:", len(not_rigid))
    print("# of undetermined clusters:", len(idk))

    print("\nFor hypostatic clusters:")
    print("# of hypostatic clusters:", len(hypostatic))
    #print("# of pre-stress stable hypostatics:", len(hypo_stress))
    #print("Hypostatic clusters:")
    #print(hypostatic)
    #print("hypostatic cluster has rigidity value", hypo_rigid)

    print("\nAvg cond of first-order R(x):", np.mean(cond_1))
    print("\nAvg cond of pre-stress R(x):", np.mean(cond_pre))
    print("\nAvg cond of |W| = 0 R(x):", np.mean(cond_w0))
    return isostatic

def test_numerical(start_n,end_n):
    d = 3
    for n in range(start_n, end_n):
        print("\nTesting n =", n)
        clusters = parse_coords(n)
        print("# of clusters:", len(clusters))
        for cluster in clusters: #cluster is x
            assert len(cluster) == d*n
            #print("cluster:", cluster)
            RA = rigidity_matrix(cluster, d, n) # returns (R, A)
            # contacts = 0
            # for i in range(len(RA[1])):
            #     for j in range(i+1, len(RA[1][0])):
            #         if RA[1][i][j] == 1:
            #             contacts += 1
            # print("cluster has # contacts:", contacts)
            rigid = is_rigid(RA,d,n) #0, 1, or 2
            if isinstance(rigid, tuple): # rigid == -1 or rigid == 0
                print("\nReturned by analytic method:", rigid[0])
                print("Calculating numerical method...")
                lenB = numerical_dim(cluster,d,n,RA[1],rigid[1])
                print("Length of estimated basis:", lenB)
                print()

def moments(clusters, n, p=2):
    d = 3
    moments = []
    for cluster in clusters:
        center = [0,0,0]
        coords = [cluster[3*i:3*i+3] for i in range(n)]
        for coord in coords: # e.g. [0,0,0]
            center[0] += (1/n) * coord[0]
            center[1] += (1/n) * coord[1]
            center[2] += (1/n) * coord[2]
        moment = 0
        for coord in coords:
            moment += sum([abs(coord[i] - center[i]) ** p for i in range(d)])
        moments.append(moment)
    return moments

def min_moment(clusters, n, p=2):
    d = 3
    min_mom = 100
    min_x = None
    for cluster in clusters:
        center = [0,0,0]
        coords = [cluster[3*i:3*i+3] for i in range(n)]
        for coord in coords: # e.g. [0,0,0]
            center[0] += (1/n) * coord[0]
            center[1] += (1/n) * coord[1]
            center[2] += (1/n) * coord[2]
        moment = 0
        for coord in coords:
            moment += sum([abs(coord[i] - center[i]) ** p for i in range(d)])
            #moment += (np.linalg.norm(np.array(coord) - np.array(center)) ** 2)
        if moment < min_mom:
            min_mom = moment
            min_x = cluster
        # if moment <= 13.71957671958:
        #     R = rigidity_matrix(cluster, 3, n)
        #     contacts = 0
        #     for i in range(len(R[1])):
        #         for j in range(i+1, len(R[1][0])):
        #             if R[1][i][j] == 1:
        #                 contacts += 1
        #     print("# contacts of min:", contacts)
        #     break
    print("min moment cluster:", min_x, "moment =", min_mom)
    return min_x

def max_contacts(n):
    clusters = parse_coords(n)
    max_conts = 0
    max_clusters = []
    for cluster in clusters: #cluster is x
        R = rigidity_matrix(cluster, 3, n)
        contacts = 0
        for i in range(len(R[1])):
            for j in range(i+1, len(R[1][0])):
                if R[1][i][j] == 1:
                    contacts += 1
        if contacts > max_conts:
            max_conts = contacts
            max_clusters = [cluster]
        elif contacts == max_conts:
            max_clusters.append(cluster)
    return max_clusters

def test_moments(p=2): # p is the pth moment, default is 2nd
    for n in range(9,10):
        print("\nn =", n)
        clusters = parse_coords(n)
        moms = moments(clusters, n, p)
        # print("avg =", sum(moms)/len(moms))
        # min_mom = min(moms)
        # print("min =", min_mom)
        # almost_min = 0
        # for mom in moms:
        #     if 0 < mom - min_mom < 0.01*min_mom:
        #         #print("almost:", mom)
        #         almost_min += 1
        # print("# of almost min =", almost_min)

        # max_clusters = max_contacts(n)
        # for i, cluster in enumerate(max_clusters):
        #     print(i+1, ":", cluster)
        # max_moments = moments(max_clusters, n, p)
        # print(max_moments)
        # print("avg of max moments =", sum(max_moments)/len(max_moments))

        min_x = min_moment(clusters, n, p)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x1s, y1s, z1s = [], [], []
        for i in range(0,len(min_x),3):
            x1s.append(min_x[i])
            y1s.append(min_x[i+1])
            z1s.append(min_x[i+2])
        ax.scatter(x1s, y1s, z1s, c='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        # Number of contacts of clusters of minimal moment.
        # R = rigidity_matrix(min_x, 3, n)
        # contacts = 0
        # for i in range(len(R[1])):
        #     for j in range(i+1, len(R[1][0])):
        #         if R[1][i][j] == 1:
        #             contacts += 1
        # print("# contacts:", contacts)

        # For each cluster with minimal p-moment report the q-moments for that cluster for q different than p.
        for q in [1/2, 1, 2, 4]:
            if p != q:
                q_moment = moments([min_x], n, q)
                print(q, 'moment =', q_moment)

        # print(moms)
        # plt.hist(moms)
        # plt.xlabel('n = ' + str(n))
        # plt.ylabel(str(p) + ' moment')
        # plt.show()

def test_similar_moments():
    n = 14
    # maybe_same1 = [-0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.0, 0.5773502691896257, 0.816496580927726, -0.0, 0.5773502691896257, -0.816496580927726, -0.5, 0.8660254037844386, 0.0, 1.0, 1.7320508075688772, -0.0, 0.0, 1.7320508075688772, 0.0, 0.5, 1.4433756729740643, 0.816496580927726, 0.5, 1.4433756729740643, -0.816496580927726, 1.5, 0.8660254037844386, -0.0, 1.0, 0.5773502691896257, 0.816496580927726, 1.0, 0.5773502691896257, -0.816496580927726, 0.5, 0.8660254037844386, -0.0]
    # maybe_same2 = [-0.0, 0.0, -0.0, 1.0, 0.0, -0.0, 0.5, 0.8660254037844386, -0.0, -0.5, 0.2886751345948129, 0.816496580927726, -0.0, -0.5773502691896257, 0.816496580927726, 1.0, 0.5773502691896257, 1.632993161855452, 1.5, 0.2886751345948129, 0.816496580927726, 0.0, 0.5773502691896257, 1.632993161855452, 1.0, 1.1547005383792515, 0.816496580927726, 0.5, -0.2886751345948129, 1.632993161855452, 1.0, -0.5773502691896257, 0.816496580927726, 0.0, 1.1547005383792515, 0.816496580927726, 0.5, 0.2886751345948129, 0.816496580927726]
    maybe_same1 = [0.5000000000000006, -0.0962250448649387, 2.1773242158072694, 0.0, 0.0, 0.0, 0.9999999999999999, -1e-16, -1e-16, 0.4999999999999999, 0.866025403784439, 0.0, 1e-16, 1.1547005383792515, 0.8164965809277264, 1.0000000000000002, 1.1547005383792515, 0.8164965809277261, 0.5000000000000002, -0.6735753140545642, 0.5443310539518172, -0.4999999999999999, 0.2886751345948125, 0.8164965809277263, 1.5000000000000002, 0.2886751345948125, 0.816496580927726, 2e-16, -0.3849001794597515, 1.3608276348795436, 1.0000000000000004, -0.3849001794597515, 1.3608276348795434, 2e-16, 0.5773502691896253, 1.632993161855452, 1.0, 0.5773502691896251, 1.632993161855452, 0.5, 0.2886751345948125, 0.816496580927726]
    maybe_same2 = [0.5000000000000003, 1.443375672974064, 1.6329931618554525, 0.0, 0.0, 0.0, 1.0, -1e-16, -2e-16, -1e-16, -0.5773502691896263, 0.8164965809277259, 1.0, -0.5773502691896263, 0.8164965809277257, 0.5000000000000001, 0.8660254037844387, 1e-16, -0.5, 0.2886751345948125, 0.8164965809277263, 1.5000000000000002, 0.2886751345948126, 0.8164965809277258, 0.5000000000000002, -0.2886751345948134, 1.632993161855452, 1e-16, 1.1547005383792515, 0.8164965809277264, 1.0000000000000002, 1.1547005383792515, 0.8164965809277261, 3e-16, 0.5773502691896253, 1.6329931618554523, 1.0000000000000004, 0.5773502691896253, 1.632993161855452, 0.5000000000000001, 0.2886751345948126, 0.8164965809277261]
    A1 = adj_matrix(maybe_same1, 3, n)
    A2 = adj_matrix(maybe_same2, 3, n)
    dict1 = adj_matrix_to_dict(A1)
    dict2 = adj_matrix_to_dict(A2)
    G1 = pynauty.Graph(n, False, dict1)
    G2 = pynauty.Graph(n, False, dict2)
    print("isomorphic?:", pynauty.isomorphic(G1, G2))
    # graph(maybe_same1, maybe_same2, 'two n=13 clusters')
    # for p in [1/2, 1 , 3/2,  2 , 4 , 5.2, 100]:
    #     print(p, 'moment =', moments([maybe_same1, maybe_same2], n, p))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1s, y1s, z1s = [], [], []
    for i in range(0,len(maybe_same1),3):
        x1s.append(maybe_same1[i])
        y1s.append(maybe_same1[i+1])
        z1s.append(maybe_same1[i+2])
    ax.scatter(x1s, y1s, z1s, c='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1s, y1s, z1s = [], [], []
    for i in range(0,len(maybe_same2),3):
        x1s.append(maybe_same2[i])
        y1s.append(maybe_same2[i+1])
        z1s.append(maybe_same2[i+2])
    ax.scatter(x1s, y1s, z1s, c='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def test_manifold(n):
    n = 6
    d = 3
    #to_break = combos(list(range(n*d - 6)),1)
    to_break = [[9]] # note: [9] runs all the way through; [6] increases dimension
    #to_break = [[x] for x in range(12)]
    no_manifold = []
    for breaks in to_break:
        clusters = parse_coords(6) # 2 clusters for n=6
        c1, c2 = clusters[0], clusters[1] # distance between = 3.3665016461206925
        # print("distance between =", np.linalg.norm(np.array(c1)-np.array(c2)))
        graph(c1, c2, 'original 2 clusters n=6', np.linalg.norm(np.array(c1)-np.array(c2)))

        # make a 1-dimensional manifold first
        R1,A1 = rigidity_matrix(c1,d,n)
        print(np.array(A1))
        # print("\nInitial null space:", linalg.null_space(R1))
        for broken_i in breaks:
            R1 = np.delete(R1,broken_i,0)
        #print("size of R after rowdel:", R1.shape)
        A1 = update_A(A1, breaks)
        print("after deleting:\n", A1)
        # print("\nNew null space:", linalg.null_space(R1))
        lenB, B = numerical_dim(c1,d,n,A1,linalg.null_space(R1))
        print("len(B) =", lenB)
        if lenB == 1:
            # print("\nsize of B =", lenB)
            # print(B)
            #manifold_breaks.append(breaks[0])
            new_c = manifold(c1,d,n,A1,breaks,B)
            print(new_c)
        else:
            print("no 1-dim manifold for breaks =", breaks)
            no_manifold.append(breaks)
    print("didn't work for", no_manifold)
    
    #print("indices that create 1d manifold:", manifold_breaks)

def test_misc():
    
    test_tree = bst()
    print("should be nothing: ", end="")
    print(test_tree.inorder())
    print(test_tree.find(4)) #shouldn't work
    test_tree.insert(4)
    print(test_tree.find(4))
    nums = [1,5,8,3,0,3,6,7]
    for num in nums:
        test_tree.insert(num)
    print(test_tree.inorder())

    clustree = bst()
    cluster1 = cluster(adjv)
    clustree.insert(cluster1)
    print(clustree.inorder())

    #test comparison operators - works
    arr1 = np.array([0,0,0,1,0,1])
    arr2 = np.array([0,1,0,1,0,1]) #arr1 < arr2
    clus1 = cluster(arr1)
    clus2 = cluster(arr2)
    print(clus1<clus2)
    
    #test pynauty
    print("\nTESTING PYNAUTY\n")
    g = Graph(6, False, {0:[1,4],1:[2,4],2:[3],3:[4,5]})
    print(g.adjacency_dict)
    aut = autgrp(g)
    print(aut)
    relabel = canon_label(g)
    print(relabel)

    #test cluster class
    print("\n\nTESTING ADJACENCY MATRIX SETUP\n")
    #adjv = adj_dict_to_vector(g)
    #print(adjv)

    test_d = 3
    test_n =6
    test_m = 12

    adjm = adj_dict_to_vector(g)
    print(adjm)

    test_A = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]

    #test_x = [random.randint(1,10) for coord in range(test_d*test_n)] #vector dimension  dn
    test_x = [0.0000000000000000,0.0000000000000000,0.0000000000000000,0.5555555555555556, \
        1.2830005981991683,0.9072184232530289,1.0000000000000000,-0.0000000000000000,-0.0000000000000000, \
        1.3333333333333333,0.7698003589195010,0.5443310539518174,0.5000000000000000,0.8660254037844386, \
        -0.0000000000000000,0.5000000000000000,0.2886751345948129,0.8164965809277260]
    
    print("test_x:",test_x)
    Rx = rigidity_matrix(test_x, test_d, test_n)
    print(Rx)

    x = [1.0925925925925926, 1.6572091060072591, 0.1512030705421715, 1.0925925925925926, 0.6949586573578829, 1.5120307054217148, -0.0, -0.0, 0.0, 1.0, 0.0, -0.0, 0.5555555555555556, 1.2830005981991683, 0.9072184232530289, 0.5, 0.8660254037844386, 0.0, 0.5, 0.2886751345948129, 0.816496580927726, 1.3333333333333333, 0.769800358919501, 0.5443310539518174]
    print("cluster:", x)
    R = rigidity_matrix(x, 3, 8)
    #print(R)
    print(is_rigid(R,3,8))

    print("\n\nTESTING SETUP FOR RIGIDITY\n")
    r = []
    for contact in range(test_m):
        r.append([random.randint(1,10) for coord in range(test_d*test_n)])
    test_r = np.array(r) #should be np 2d array
    test_r = constrain(test_r, test_d, test_n)
    print(test_r)
    print(is_rigid(test_r, test_d, test_n))

    # test if sign_def() works: it does work
    pos_eigs = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    not_def = np.array([[9,0,-8],[6,-5,-2],[-9,3,3]])
    print("should be T for positive definite:", sign_def(pos_eigs))
    print("should be F for this one:", sign_def(not_def))

    print("\nTest projection") # must pass in np arrays - so check for this
    b=np.array([1,2,2])
    A = np.array([[1,1],[1,2],[1,3]]).T
    #b = [1,2,2]
    # A = [[1,1],[1,2],[1,3]]
    print(projection(b,A))

    A = np.array([[1,1],[1,2],[1,3]])
    b = b[:,None]
    x, res, rank, s, = np.linalg.lstsq(A,b)
    print(np.matmul(A,x))

    hypo_sample = [0.0, 0.0, 0.0, 1.0, 1e-16, 1e-16, -0.5000000000000001, 0.8660254037844386, 7.2894146e-09, 1.0000000033065455, 1.6037507496579957, 0.4536092048770559, 0.9999999940482179, 0.5773502657533626, -0.8164965833575311, -3.9678548e-09, 1.5396007262783127, -0.5443310398639957, 3.3065458e-09, 1.603750740716201, 0.4536092267089183, 0.999999996032145, 1.53960071554816, -0.5443310604312973, 1.5000000000000002, 0.8660254037844389, -7.2894148e-09, 0.5, 0.8660254037844386, -2e-16]
    RA = rigidity_matrix(hypo_sample,3,10)
    R, A = RA
    u, s, vh = np.linalg.svd(R, full_matrices=True)
    print(s)

    print("\nRIGHT NULL\n")
    print(is_rigid(RA,3,10))
    right_null = linalg.null_space(R)
    lenB = numerical_dim(hypo_sample,3,10,A,right_null)
    print("Length of estimated basis:", lenB)
    
    A = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]
    contacts = 0
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            if A[i][j] == 1:
                contacts += 1
    print("contacts:", contacts)


if __name__ == '__main__':
    
    #test_misc()

    # print("\n\nTESTING FOR RIGIDITY\n")

    # print("\n\nTEST CLUSTERS n=6 THROUGH 10\n")
    # n = 11
    # hypos = test_hc_rigid_clusters(n, n+1)
    # print("# hypostatic =", len(hypos))
    # dimensions = []
    # for cluster in hypos:
    #     R, A = rigidity_matrix(cluster, 3, n)
    #     dimensions.append(numerical_dim(cluster, 3, n, A, linalg.null_space(R))[0])
    # print("\nDIMENSIONS FOR N=" + str(n) + " ISOSTATICS:")
    # print(dimensions)

    # len0, len1, len2, failed = 0, 0, 0, 0
    # for dim in dimensions:
    #     if dim == 0:
    #         len0 += 1
    #     elif dim == 1:
    #         len1 += 1
    #     elif dim == 2:
    #         len2 += 1
    #     elif dim == -1:
    #         failed += 1
    #     else:
    #         raise RuntimeError(">:C")
    # print("# where len(B)=0:", len0)
    # print("# where len(B)=1:", len1)
    # print("# where len(B)=2:", len2)
    # print("# where Newton's didn't converge:", failed)

    # print("\nTest a non-rigid cluster!")
    # test_hypercube()

    # print("\nTest other rigid structures:")
    # test_simplex()

    # print("\nTest numerical method")
    # test_numerical(9, 10)


    # print("\nManifold algorithm:")
    # test_manifold(0)
    #print(combos(list(range(12)),2))

    print("\nTesting moments:")
    test_moments(1)
    test_moments(2)
    test_moments(4)

    # print("\nSemidefinite testing:")
    # pss1 = [1.0925925925925926, 1.6572091060072591, 0.1512030705421715, 1.0925925925925926, 0.6949586573578829, 1.5120307054217148, -0.0, -0.0, 0.0, 1.0, 0.0, -0.0, 0.5555555555555556, 1.2830005981991683, 0.9072184232530289, 0.5, 0.8660254037844386, 0.0, 0.5, 0.2886751345948129, 0.816496580927726, 1.3333333333333333, 0.769800358919501, 0.5443310539518174]
    # pss2 = [0.5, 1.4433756729740643, 0.816496580927726, 0.5, 0.2886751345948129, 1.632993161855452, -0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.5, 0.8660254037844386, 0.0, 0.5, -0.2886751345948129, 0.816496580927726, 0.0, 0.5773502691896257, 0.816496580927726, 1.0, 0.5773502691896257, 0.816496580927726]
    # pss3 = [-0.0, 0.0, -0.0, 1.0, 0.5773502691896257, 1.632993161855452, 1.5, 0.8660254037844386, 0.0, 1.0, 0.0, -0.0, 0.5, 0.8660254037844386, 0.0, 1.5, 0.2886751345948129, 0.816496580927726, 1.0, 1.1547005383792515, 0.816496580927726, 0.5, 0.2886751345948129, 0.816496580927726]
    # R, A = rigidity_matrix(pss1, 3, 8)
    # left_null = linalg.null_space(R.T).T
    # # print('left null:', left_null)
    # omega = omega(left_null[0], A, 3)
    # print("shape of omega:", omega.shape)
    # print(omega)
    
    # c1 = [1.3950617283950617, -0.1140444976177039, -1.0080204702811433, -0.2962962962962963, 0.8553337321327789, -0.604812282168686, 0.0, 0.0, 0.0, 0.5720164609053497, -0.1330519138873212, -1.7203549359464845, 1.0, -0.0, 0.0, 0.0987654320987654, 0.741289234515075, -1.6128327524498292, 1.3333333333333333, 0.769800358919501, -0.5443310539518174, 0.5555555555555556, 1.2830005981991683, -0.9072184232530289, 0.5, 0.8660254037844386, 0.0, 1.0925925925925926, 0.6949586573578829, -1.5120307054217148, 0.5, 0.2886751345948129, -0.816496580927726]
    # c2 = [-0.2962962962962963, 0.8553337321327789, 0.604812282168686, 0.0987654320987654, 0.741289234515075, 1.6128327524498292, -0.0, -0.0, 0.0, 0.6584362139917695, -0.1900741626961731, 1.6800341171352386, 1.0, -0.0, 0.0, 1.3950617283950617, -0.1140444976177039, 1.0080204702811433, 0.5555555555555556, 1.2830005981991683, 0.9072184232530289, 0.5, 0.8660254037844386, 0.0, 1.0925925925925926, 0.6949586573578829, 1.5120307054217148, 1.3333333333333333, 0.769800358919501, 0.5443310539518174, 0.5, 0.2886751345948129, 0.816496580927726]
    # norm = 0
    # norm_vec = []
    # for i in range(11):
    #     x = np.array(c1[3*i:3*i+3])
    #     y = np.array(c2[3*i:3*i+3])
    #     this_norm = np.linalg.norm(x-y)
    #     norm += this_norm
    #     norm_vec.append(this_norm)
    # print("\nTOTAL NORM =", norm)
    # print("\nNORM =", norm_vec)

    
    