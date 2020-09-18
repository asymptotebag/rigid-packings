from pynauty import * #please figure out how to use this
import numpy as np
import random

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
    #return adj_matrix
    return adj_matrix.flatten()

#convert adjacency matrix to system of equations (1)
'''
def add_cluster(cluster): #i guess cluster will be a cluster object
    pass
'''

def constrain(matrix, d, n):
    '''
    Adds rows of constrained vertices to the end of the rigidity matrix, a 2d numpy array. 
    Should mutate input matrix. (helper function)
    Returns constrained matrix.
    '''
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
    print("TESTING PYNAUTY\n")
    g = Graph(6, False, {0:[1,4],1:[2,4],2:[3],3:[4,5]})
    print(g.adjacency_dict)
    aut = autgrp(g)
    print(aut)
    relabel = canon_label(g)
    print(relabel)

    #test cluster class
    print("\n\nTESTING ADJACENCY STRUCTURES\n")
    adjv = adj_dict_to_vector(g)
    print(adjv)

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

    #test rigidity setup
    print("\n\nTESTING SETUP FOR RIGIDITY\n")
    test_d = 5
    test_n = 10
    test_m = 4

    r = []
    for contact in range(test_m):
        r.append([random.randint(1,10) for coord in range(test_d*test_n)])
    test_r = np.array(r) #should be np 2d array
    test_r = constrain(test_r, test_d, test_n)
    print(test_r)
        