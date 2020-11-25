from setup import *

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
    return hypostatic

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
    print("\nA =", np.array(A1))
    print(R1)
    print("\nInitial null space:", linalg.null_space(R1))
    print("\nLeft null space:", linalg.null_space(R1.T))
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
