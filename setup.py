from pynauty import * #please figure out how to use this
import numpy as np
import random
from scipy import linalg
#from numpy.linalg import matrix_rank

# GLOBAL VARIABLES
ADJACENCY_TOL = 10**-5

#convert adjacency dictionary (which is already the adj matrix in a hash table)
# to a binary vector

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

# trying out some functions here ************************

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

'''
def add_cluster(cluster): #i guess cluster will be a cluster object
    pass
'''

def rigidity_matrix(x, d, n):
    # can iterate through adjacency matrix A and look for 1s (indicate contacts)
    # shouldn't depend on the adjacency matrix ********************
    #kd tree
    #tolerance for adjacency = 10^-3 or 10^-5
    #A = [[]] #NEED TO DEFINE THIS

    # CREATE ADJACENCY MATRIX FROM VECTOR X
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
        print("new row =", new_row)
        A.append(new_row)
    
    #assert len(A) == n
    #assert len(A[0]) == n
    #why is it returning an identity matrix???
    print(A)

    R = []
    print("\n")
    for i in range(len(A)):
        print("i =",i)
        for j in range(i+1, len(A[0])): 
            print("\nj =",j)
            #i+1 to skip the diagonal (sphere can't contact itself so diagonal always 0)
            if A[i][j] == 1:
                #i is always less than j b/c we only iterate through upper triangle
                new_row = []
                prezeros = d*i
                midzeros = d*(j-i-1)
                postzeros = d*(len(A[0])-j-1)
                iminj = [] #p_ik - p_jk
                jmini = [] #-(p_ik-p_jk)
                print("d*i =",d*i,"\t(d+1)*i =",d*i+d)
                print("d*j =",d*j,"\t(d+1)*j =",d*j+d)
                print("coordinates of i:", x[d*i:d*i+d])
                print("coordinates of j:", x[d*j:d*j+d])
                for coord_i, coord_j in zip(x[d*i:d*i+d], x[d*j:d*j+d]):
                    iminj.append(coord_i - coord_j)
                    jmini.append(coord_j - coord_i)

                print("iminj =", iminj)
                print("jmini =", jmini)

                new_row += [0 for zero in range(prezeros)]
                new_row += iminj
                new_row += [0 for zero in range(midzeros)]
                new_row += jmini
                new_row += [0 for zero in range(postzeros)]
                
                print("new row is", len(new_row), "elements long")
                #why is the length of new_row different each time??
                #assert len(new_row) == n*d

                R.append(new_row)
    
    R = constrain(R, d, n)
    return R

def constrain(matrix, d, n):
    '''
    Adds rows of constrained vertices to the end of the rigidity matrix, a 2d numpy array. 
    Should mutate input matrix. (helper function)
    Returns constrained matrix.
    '''
    matrix = np.array(matrix)
    #print(matrix)
    #test d = 5, n = 10, generate numpy array
    s_j = [] #indices/vertices to constrain
    ind = 0
    for sphere in range(d):
        ind += sphere
        s_j += list(range(ind, d*(sphere+1)))
        ind = d*(sphere+1)
    
    print("constrained vertices:", s_j) #should be 1,2,3,4,5,7,8,9,10,13,14,15,19,20,25

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

#trying the rigidity test here??
def is_rigid(R, d, n): #R is rigidity matrix (should be 2d numpy array)
    '''
    Returns 1 if cluster if 1st order rigid, 2 if it is pre-stress stable.
    Returns 0 (maybe) if the cluster is not rigid.
    '''
    #TEST FIRST ORDER RIGIDITY
    # if right null space V, dim = n_v = 0 --> return 1 (for 1st order rigid)
    right_null = linalg.null_space(R) #V, gives orthonormal basis
    print("basis of right null space:", right_null)
    n_v = right_null.shape[1]
    print("dim(right null space) =", n_v)
    
    if n_v == 0: #(N,K) array where K = dimension of effective null space
        print("First-order rigid")
        return 1
    

    #TEST PRE-STRESS STABILITY
    left_null = linalg.null_space(R.T)#W
    n_w = left_null.shape[1]
    if n_w == 0:
        print("Not rigid")
        return 0 #???
    else: 
        for m in range(n_w): #iterate from 1 to n_w
            b = [0 for zero in range(n_v)] #b has size n_v
            b[m] = 1 #b_m = e_m
            #now make Q and do b*Q
            Q_m = [] #dimension n_v x n_v
            for i in range(n_v):
                new_row = []
                for j in range(n_v):
                    wR = np.matmul(left_null[m].T, rigidity_matrix(right_null[i], d, n))
                    # does it need to be the transpose of right_null[i]?????????
                    new_row.append(np.matmul(wR, right_null[j]))
                Q_m.append(new_row)
            if sign_def(Q_m):
                print("Pre-stress stable")
                return 2
    print("Unable to determine rigidity")
    return 0 #????

if __name__ == '__main__':
    '''
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
    '''

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

    #test_A = [[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]

    #test_x = [random.randint(1,10) for coord in range(test_d*test_n)] #vector dimension  dn
    #test_x = [0.0000000000000000,0.0000000000000000,0.0000000000000000,0.5555555555555556, \
        #1.2830005981991683,0.9072184232530289,1.0000000000000000,-0.0000000000000000,-0.0000000000000000, \
        #1.3333333333333333,0.7698003589195010,0.5443310539518174,0.5000000000000000,0.8660254037844386, \
        #-0.0000000000000000,0.5000000000000000,0.2886751345948129,0.8164965809277260]
    test_x = [0.0000000000000000,   0.0000000000000000,   0.0000000000000000,   1.0000000000000000, \
          0.0000000000000000,   0.0000000000000000,   0.5000000000000000,   0.8660254037844386,  \
               0.0000000000000000,   0.5000000000000000,  -0.2886751345948129,  -0.8164965809277260, \
                    -0.0000000000000000,   0.5773502691896257,  -0.8164965809277260,   1.0000000000000000,\
                           0.5773502691896257,  -0.8164965809277260 ]
    #print("test_x:",test_x)
    Rx = rigidity_matrix(test_x, test_d, test_n)
    print(Rx)

    print("\n\nTESTING FOR RIGIDITY\n")
    print(is_rigid(Rx, test_d, test_n))

    # test if sign_def() works: it does work
    pos_eigs = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    not_def = np.array([[9,0,-8],[6,-5,-2],[-9,3,3]])
    print("should be T for positive definite:", sign_def(pos_eigs))
    print("should be F for this one:", sign_def(not_def))

    '''
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
    '''

    #test rigidity setup
    '''
    print("\n\nTESTING SETUP FOR RIGIDITY\n")
    r = []
    for contact in range(test_m):
        r.append([random.randint(1,10) for coord in range(test_d*test_n)])
    test_r = np.array(r) #should be np 2d array
    test_r = constrain(test_r, test_d, test_n)
    print(test_r)
    print(is_rigid(test_r, test_d, test_n))
    '''
        