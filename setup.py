import os
import math
import random
import numpy as np
from scipy import linalg
from scipy import optimize
from pynauty import * #please figure out how to use this
import cvxpy as cvx
#from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
ADJACENCY_TOL = 10**-5
INITIAL_ADJACENCY_TOL = 1e-3
SAME_COORDS_TOL = 1e-5

INITIAL_STEP_SIZE = 5e-2
STEP_SIZE = 5e-3
TOL_NEWTON = 9e-16 # doesn't this seem really small?
MAX_STEP_NEWTON = 0.02
X_TOL_MAX = 10*STEP_SIZE
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

def adj_matrix(x,d,n, A = [], returnA = True):
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
                if i!=j and norm - 1 <= ADJACENCY_TOL: #otherwise it will think each sphere contacts itself
                    new_row.append(1)
                else:
                    new_row.append(0)
            #print("new row =", new_row)
            A.append(new_row)
    return A

def system_eqs(x, d, n, A):
    '''
    Given cluster coordinates x (vector sized nd) and its adjacency matrix, return a vector with elements
    corresponding to all contacts (where A[i][j]=1) and whether the current coordinates (since this is used
    when x has been perturbed) still fulfill the original equation. Element is 0 if a contact still holds; 
    otherwise, it is the distance between them. 
    *** Should an error be raised if the distance is negative? (spheres can't intersect)
    '''
    distances = []
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
                if norm - 1 < 1e-6: 
                    distances.append(0) # contact is still present
                else:
                    distances.append(norm - 1)
    distances += [0 for constraint in range(int(d*(d+1)/2))]
    return np.array(distances)
    #return np.array(distances).T

def rigidity_matrix(x, d, n, A = [], returnA = True):
    # can iterate through adjacency matrix A and look for 1s (indicate contacts)
    # shouldn't depend on the adjacency matrix ********************
    #kd tree
    #tolerance for adjacency = 10^-3 or 10^-5
    
    # CREATE ADJACENCY MATRIX FROM VECTOR X
    A = adj_matrix(x, d, n, A, returnA)
    
    #assert len(A) == n
    #assert len(A[0]) == n
    #why is it returning an identity matrix???
    #print(A)

    R = []
    #print("\n")
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
                return False
    elif eigs[0] < 0: #test negative sign definite
        for eig in eigs:
            if eig >= 0:
                return False
    else: #first term 0
        return False
    return True

def is_rigid(RA, d, n): #R is rigidity matrix (should be 2d numpy array)
    '''
    Returns 1 if cluster if 1st order rigid, 2 if it is pre-stress stable.
    Returns 0 (maybe) if the cluster is not rigid.
    '''
    #print("RA is " + str(len(RA)) + " elements long.")
    R, A = RA
    #print(R)
    #print(A)

    #TEST FIRST ORDER RIGIDITY
    # if right null space V, dim = n_v = 0 --> return 1 (for 1st order rigid)
    #print("dimension of R:", R.shape[0], "x", R.shape[1])
    right_null = linalg.null_space(R) #V, gives orthonormal basis
    '''
    instead of null_space(), check min singular value, if nonzero, kernel is empty
    '''
    #print("basis of right null space:\n", right_null)
    n_v = right_null.shape[1]
    #print("shape of right null =", right_null.shape)
    #print(right_null.T)
    #print("dim(right null space) =", n_v)
    
    if n_v == 0: #(N,K) array where K = dimension of effective null space
        print("First-order rigid")
        return (1,right_null)
    

    #TEST PRE-STRESS STABILITY
    transpose = R.T
    #print("TRANSPOSE:", transpose)
    left_null = linalg.null_space(transpose)#W
    print("SHAPE:", left_null.shape)
    n_w = left_null.shape[1]
    #print("\nbasis of left null space:", left_null)
    #print("dim(left null space) =", n_w)
    if n_w == 0: # this seems to mean the cluster is hypostatic
        print("Not rigid")
        print("Dimension =", n_v) # for hypostatic clusters, D = n_v = 1
        return (0,right_null) #??? return n_v
    else: 
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
            if sign_def(Q_m):
                print("Pre-stress stable")
                return (2,right_null)
    print("Unable to determine rigidity")
    # numerical_dimension(x, d, n, A, right_null)
    # does x need to be passed into this function as well??
    return (-1,right_null) #????

def omega(w):
    '''
    Returns stress matrix Omega with dimensions dn x dn.
    Omega acts on two flexes (u,v in right kernel) as u.T*Omega*v = sum( w_ij (u_i - u_j)(v_i - v_j)) - in R
    '''
    raise NotImplementedError

def sdp(V, W, omega):
    # M_i = V.T * Omega(w_i) * V
    n_v = V.shape[1]
    n_w = W.shape[1]
    I = np.identity(n_v)

    t = cvx.Variable() # scalar
    a = cvx.Variable(n_w)
    X = cvx.Variable((n_v,n_v),symmetric=True)
    constraints = [X >> 0, cvx.norm(a)<=1]
    for i in range(n_w):
        M = np.matmul(V.T, omega(W[i]))
        M = np.matmul(M, V)
        constraints += [] # add the constraints
    #constraints.append()

    prob = cvx.Problem(cvx.Maximize(t),constraints)
    prob.solve()
    t_opt = prob.value
    a_opt = a.value
    raise NotImplementedError

def numerical_dim(x, d, n, A, right_null): # if is_rigid returned 0 or -1
    '''
    x is the coordinates of the cluster (the "approximate solution").
    right_null is the basis of the right null space of R(x), which was determined and passed in via
    is_rigid(). right_null must be nonempty (or this function would not have been called).
    Return estimated dimension of cluster.
    '''
    # print(right_null) 
    failed_newtons = []
    basis = []
    for v_j in right_null.T: # extracts the "vertical" basis vectors
        #print("v_j =", v_j)
        for sign in ['+','-']:
            # take step in both directions
            if sign == '+':
                x_plus = [coord + INITIAL_STEP_SIZE*v for coord, v in zip(x, v_j)]
            else:
                x_plus = [coord - INITIAL_STEP_SIZE*v for coord, v in zip(x, v_j)]

            # project onto constraints
            #print("x =", x)
            #print("xplus =", x_plus)
            projx_plus, iters_plus = newtons(system_eqs, rigidity_matrix, x_plus, d, n, A) #hello what are F and J????
            
            print("iterations =", iters_plus)
            print(system_eqs(x_plus,d,n,A))
            if iters_plus == -1:
                # exceeded maximum iterations
                failed_newtons.append((v_j, 'exceeded iters on ' + sign))
                break
            if np.inner(projx_plus - x_plus, v_j) > ORTH_TOL: 
                # i have no idea if this is the right tolerance value to use!!!
                # require (proj_x - x_plus) perpendicular to v_j --> inner product is zero
                failed_newtons.append((v_j, 'newton result was not orth for ' + sign))

            # projx_plus = optimize.root(system_eqs, x_plus, args=(x,d,n,[],True), method='krylov', jac=rigidity_matrix, tol=TOL_NEWTON, callback=None, options=None)

            # assuming x, projx, are both np vectors/arrays:
            tanv_plus = projx_plus - x
            norm_plus = np.linalg.norm(tanv_plus)
            if norm_plus > X_TOL_MAX or norm_plus < X_TOL_MIN:
                pass # reject vector (aka do nothing / skip it)
            else:
                # project onto current estimate of B
                # projection matrix P = A(A.T A)^-1 A.T
                if len(basis) == 0:
                    print("size of added:", (tanv_plus/np.linalg.norm(tanv_plus)).shape)
                    basis.append(tanv_plus/np.linalg.norm(tanv_plus))
                else:
                    # print("tanv_plus =", tanv_plus)
                    proj = projection(tanv_plus,basis).T # this probably needs to be basis.T
                    # print("proj =", proj)
                    # print("proj.T =", proj.T)
                    orthonorm = tanv_plus - proj # orthogonal portion to the projection - check this
                    if np.linalg.norm(orthonorm) > ORTH_TOL:
                        print("size:", (orthonorm/np.linalg.norm(orthonorm)).shape)
                        basis.append(orthonorm/np.linalg.norm(orthonorm))
    #print("basis =", basis)
    return (len(basis), basis)

# optimize.newton(func, x0, fprime=None, args=(), tol=TOL_NEWTON, maxiter=50, fprime2=None, 
#   x1=None, rtol=0.0, full_output=False, disp=True)

# optimize.root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)

def projection(v, A):
    '''
    Projects vector v onto the column space of A (using the SVD of A).
    '''
    A = np.array(A).T
    print("dimensions of A:", A.shape)
    
    # if len(A.shape) == 1: # basis only has 1 vector in it
    #     A_T = A[:,None] # transpose of single vector
    # else:
    #     A_T = A.T
    # ata = np.matmul(A, A_T)
    # print("ata:", ata)
    # # CHECK IF ATA IS SQUARE!!! if not use moore penrose inv - pinv
    # itmd = np.matmul(A_T,np.linalg.inv(ata)) #intermediate
    # proj_matrix = np.matmul(itmd, A)
    # project = np.matmul(proj_matrix, v)
    v = v[:,None]
    print("dimensions of v:", v.shape)
    x, res, rank, s, = np.linalg.lstsq(A,v,rcond=None)
    project = np.matmul(A, x)
    return project

def newtons(F, J, x, d, n, A, eps=TOL_NEWTON): 
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F. Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    Step size per iteration limited by MAX_STEP_NEWTON.
    """
    F_value = F(x, d, n, A) # need F(x, d, n, A)
    print("F_value =", F_value)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    print("l2 norm of vector =", abs(F_norm))
    while abs(F_norm) > eps and iteration_counter < 100:
        jac = J(x, d, n, A, False)
        #print("size:", jac.shape)
        if jac.shape[0] == jac.shape[1]: #square matrix
            delta = np.linalg.solve(jac, -F_value)
        else:
            inv = np.linalg.pinv(jac)
            delta = np.matmul(inv,-F_value)
        # constrain maximum step size (did i do this right?????????)
        delt_norm = np.linalg.norm(delta)
        if delt_norm > MAX_STEP_NEWTON:
            delta = delta*(MAX_STEP_NEWTON/delt_norm) # scale to max step size
        x = x + delta
        F_value = F(x, d, n, A)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    return x, iteration_counter

# MOVING ALONG ONE-DIMENSIONAL MANIFOLD

def within_tol(x,d,n,A,broken,tol=INITIAL_ADJACENCY_TOL):
    '''
    Return bool based on whether 2 spheres are now adjacent (??)
    broken is a list of indices that correspond to broken contacts (should only be 1 at a time).
    '''
    contacts = system_eqs(x,d,n,A)
    for i in range(len(contacts)):
        if i in broken and contacts[i] < tol:
            return i # so return index corresponding to which thing was re-contacted?
    return None #nothing is within tolerance (default)


def manifold(x0, d, n, A, basis):
    '''
    Method called only if we have a 1D solution set.
    Input: basis is np matrix B, calculated in numerical dimension method.
    '''
    x = x0 # make a copy since we're mutating x
    dirs = [] # directions for which contacts increase - may change to set
    broken = {} # map {contact # : distance}
    contacts = system_eqs(x,d,n,A) # some should be != 0
    for i in range(len(contacts)):
        if contacts[i] != 0:
            broken[i] = contacts[i]
    #for vec in basis.T:
    for vec in basis:
        new_contacts = system_eqs(x + INITIAL_STEP_SIZE*vec,d,n,A)
        increased = True
        for contact in broken:
            if new_contacts[contact] <= broken[contact]:
                increased = False
                break
        if increased:
            dirs.append(vec) # added only if the direction makes ALL contacts increase
    
    
    for v_k in dirs:
        v = v_k # update this as you go
        prev_step = x # used for when we backtrack 1 step at the end
        step_count = 0
        # create new for loop here / or while loop testing for tolerance btwn spheres
        while within_tol(x,d,n,A,broken) is None: # stop when you get 2 spheres close enough
            print("in first loop")
            step_count += 1

            # take step in tangent direction, along manifold
            step = [coord + 2*INITIAL_STEP_SIZE*v for coord, v in zip(x, v)] # is it vector x we're adding to?
            
            # project back onto manifold
            step, iters = newtons(system_eqs, rigidity_matrix, step, d, n, A)
            if iters == -1: #newton's failed
                raise KeyboardInterrupt
            newR = rigidity_matrix(step, d, n, A, False)
            #newR = rigidity_matrix(step, d, n)
            right_null = linalg.null_space(newR) # if right_null = [] the dimension decreased: should break
            print("right null:", right_null)
            print("v:", v)
            v = projection(v, right_null)
            prev_step = x # save prev position
            x = step # update current position

            #check dimension after step 1
            if step_count == 1:
                dim = is_rigid((newR, A), d, n)
                if dim[0] == 0 or dim[0] == -1: # analytic method didn't work
                    dim = numerical_dim(x, d, n, A, dim[1])
                else:
                    dim = dim[0] # only keep the number
                    if dim == 1 or dim == 2:
                        dim = 0
                    # else:
                    #     dim = 1 # ????
                # check if dim has increased or decreased
                if dim > 1:
                    return -1 # stop moving, dim increased
                elif dim < 1:
                    break # does this mean we moved back to the original??

        x = prev_step # back up 1 step

        # repeat continuation with smaller step size
        while within_tol(x,d,n,A,broken) is None: # stop when you get 2 spheres close enough, again
            print("doing final round")
            #step_count += 1
            # take step in tangent direction, along manifold
            step = [coord + STEP_SIZE*v for coord, v in zip(x, v)]
            
            # project back onto manifold
            step, iters = newtons(system_eqs, rigidity_matrix, step, d, n, A)
            if iters == -1: #newton's failed
                raise KeyboardInterrupt
            newR = rigidity_matrix(step, d, n, A, False)
            right_null = linalg.null_space(newR)
            v = projection(v, right_null)
            #prev_step = x
            x = step # update current position
        
        # Then, we project onto this new set of constraints and check if the cluster is rigid, 
        # using a new tolerance tolA to determine whether two spheres are adjacent.
        proj_final, iters = newtons(system_eqs, rigidity_matrix, x, d, n, A)
        if iters == -1: # projection failed
            '''
            delete subsets of the new constraints until the projection succeeds
            '''
            pass
        finalR = rigidity_matrix(proj_final,d,n,A)
        final_rigidity = is_rigid(finalR,d,n)
        # implement numerical method here?
        if final_rigidity == 1 or final_rigidity == 2:
            # new cluster found!
            # change implementation later
            return proj_final


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
    hypo_stress = [] #hypostatic & pre-stress (should be all of them?)
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
            #assert rigid != 0 # these all should be rigid
            #print("\n")

    print("For all clusters n =", start_n, "through", end_n-1)
    print("# of first-order rigid clusters:", len(first_rigid))
    print("# of pre-stress stable clusters:", len(pre_stress))
    print("# of non-rigid clusters:", len(not_rigid))
    print("# of undetermined clusters:", len(idk))

    print("\nFor hypostatic clusters:")
    print("# of hypostatic clusters:", len(hypostatic))
    print("# of pre-stress stable hypostatics:", len(hypo_stress))
    print("Hypostatic clusters:")
    print(hypostatic)
    #print("hypostatic cluster has rigidity value", hypo_rigid)

    print("\nAvg cond of first-order R(x):", np.mean(cond_1))
    print("\nAvg cond of pre-stress R(x):", np.mean(cond_pre))
    print("\nAvg cond of |W| = 0 R(x):", np.mean(cond_w0))

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

def moments(n):
    clusters = parse_coords(n)
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
            moment += 4 * (np.linalg.norm(np.array(coord) - np.array(center)) ** 2)
        # # specifically for n = 11:
        # if moment < 37.84:
        #     print("\nMin second moment cluster!")
        #     print(cluster)
        #     print("moment =", moment)
        moments.append(moment)
    return moments

def test_manifold(n):
    n = 6
    d = 3
    clusters = parse_coords(6) # 2 clusters for n=6
    c1, c2 = clusters[0], clusters[1]
    # make a 1-dimensional manifold first
    R1,A1 = rigidity_matrix(c1,d,n)
    # print(R1)
    # print("\nInitial null space:", linalg.null_space(R1))
    R1 = np.delete(R1,0,0)
    # print("\nAfter row del:", R1)
    # print("\nNew null space:", linalg.null_space(R1))
    lenB, B = numerical_dim(c1,d,n,A1,linalg.null_space(R1))
    print("\nsize of B =", lenB)
    print(B)
    new_c = manifold(c1,d,n,A1,B)
    print(new_c)

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

    test_x = [random.randint(1,10) for coord in range(test_d*test_n)] #vector dimension  dn
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


if __name__ == '__main__':
    
    #test_misc()

    # print("\n\nTESTING FOR RIGIDITY\n")

    # # test if sign_def() works: it does work
    # pos_eigs = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    # not_def = np.array([[9,0,-8],[6,-5,-2],[-9,3,3]])
    # print("should be T for positive definite:", sign_def(pos_eigs))
    # print("should be F for this one:", sign_def(not_def))


    print("\n\nTEST CLUSTERS n=6 THROUGH 10\n")
    #test_hc_rigid_clusters(6,10)

    print("\nTest a non-rigid cluster!")
    #test_hypercube()

    print("\nTest other rigid structures:")
    #test_simplex()

    print("\nTest projection") # must pass in np arrays - so check for this
    b=np.array([1,2,2])
    # A = np.array([[1,1],[1,2],[1,3]]).T
    # #b = [1,2,2]
    # # A = [[1,1],[1,2],[1,3]]
    # print(projection(b,A))

    # A = np.array([[1,1],[1,2],[1,3]])
    # b = b[:,None]
    # x, res, rank, s, = np.linalg.lstsq(A,b)
    # print(np.matmul(A,x))

    test_manifold(0)

    # print("\nSecond moments of clusters:")
    # for n in range(11,12):
    #     print("\nSpheres for n =", n)
    #     moms = moments(n)
    #     #print(moms)
    #     #plt.plot(moms)
    #     plt.hist(moms)
    #     plt.xlabel('n = ' + str(n))
    #     plt.ylabel('second moment')
    #     #plt.show()
    #     print("min*4 =", min(moms))
    #     hist = [x for x in moms if 37.83<x<37.9]
    #     print(hist)
    
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

    print("\nTest numerical method")
    # test_numerical(9, 10)

    # hypo_sample = [0.0, 0.0, 0.0, 1.0, 1e-16, 1e-16, -0.5000000000000001, 0.8660254037844386, 7.2894146e-09, 1.0000000033065455, 1.6037507496579957, 0.4536092048770559, 0.9999999940482179, 0.5773502657533626, -0.8164965833575311, -3.9678548e-09, 1.5396007262783127, -0.5443310398639957, 3.3065458e-09, 1.603750740716201, 0.4536092267089183, 0.999999996032145, 1.53960071554816, -0.5443310604312973, 1.5000000000000002, 0.8660254037844389, -7.2894148e-09, 0.5, 0.8660254037844386, -2e-16]
    # RA = rigidity_matrix(hypo_sample,3,10)
    # R, A = RA
    # u, s, vh = np.linalg.svd(R, full_matrices=True)
    # print(s)

    # print("\nRIGHT NULL\n")
    # print(is_rigid(RA,3,10))
    # right_null = linalg.null_space(R)
    # lenB = numerical_dim(hypo_sample,3,10,A,right_null)
    # print("Length of estimated basis:", lenB)
    
    # A = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]
    # contacts = 0
    # for i in range(len(A)):
    #     for j in range(i+1, len(A[0])):
    #         if A[i][j] == 1:
    #             contacts += 1
    # print("contacts:", contacts)
    

    # clustree = bst()
    # cluster1 = cluster(adjv)
    # clustree.insert(cluster1)
    # print(clustree.inorder())

    # #test comparison operators - works
    # arr1 = np.array([0,0,0,1,0,1])
    # arr2 = np.array([0,1,0,1,0,1]) #arr1 < arr2
    # clus1 = cluster(arr1)
    # clus2 = cluster(arr2)
    # print(clus1<clus2)
 

    # print("\n\nTESTING SETUP FOR RIGIDITY\n")
    # r = []
    # for contact in range(test_m):
    #     r.append([random.randint(1,10) for coord in range(test_d*test_n)])
    # test_r = np.array(r) #should be np 2d array
    # test_r = constrain(test_r, test_d, test_n)
    # print(test_r)
    # print(is_rigid(test_r, test_d, test_n))
    