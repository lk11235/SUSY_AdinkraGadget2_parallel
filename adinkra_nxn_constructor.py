# ******************************************************************************
# Name:    Testing
# Author:  Lucas Kang and Vadim Korotkikh
# Email:   lucas_kang@brown.edu and va.korotki@gmail.com

from multiprocessing import Pool
from cython.parallel import *
from numpy.linalg import inv
from numpy.core.umath_tests import inner1d
from fractions import Fraction as F
import numpy as np
import itertools
import os
import time
import sys
import time
from tqdm import *
import re
import gc

import matplotlib.pyplot as plt
import collections

os.system("taskset -p 0xff %d" % os.getpid())

# Makes colums/rows that compose an Identity matrix
def unit_vector(n,i):
    vec     = [0 for j in range(n)]
    vec[i]  = 1
    return np.array(vec)

# Creates the 24 unsigned permutation matrices from permutation group S4
def gen_permutations(n):
    # bracket_n = list(range(n))
    perms_list = list(itertools.permutations(range(n), n))
    ts = [unit_vector(n,i) for i in range(n)]
    temp = []
    for mi in perms_list:
        columns = [ts[mi[i]] for i in range(n)]
        temp_mat = np.column_stack(columns)
        matint = temp_mat.astype(int)
        temp.append(matint)
    return temp

# Generate all sign permutations of an nxn Identity Matrix
def gen_signm(n):
    items   = [1] * int(n)
    # itemsl = np.ones(n)
    # items = itemsl.tolist()
    # n   = int(n)
    # items = [1] * n
    sign_mat = []
    for signs in itertools.product([-1,1], repeat=n):
        temp = np.array([a*sign for a,sign in zip(items,signs)],dtype=int)
        # ptemp.append(temp)
        sign_mat.append(np.diag(temp))
    return sign_mat

# Creates the (384) sign permutation matrices
def gen_product_matrices(n):
    legal_matrices = []
    # from adinkra_tetrad_calc import gen_signm
    sign_pmat = gen_signm(n)
    uperm_mat = gen_permutations(n)
    for x in sign_pmat:
        # t1 = np.asmatrix(x)
        for y in uperm_mat:
            # t2 = np.asmatrix(y)
            # legal_matrices.append(np.matmul(t1,t2))
            legal_matrices.append(np.dot(x,y))
    return legal_matrices

# ****************
def pairing(matli, matlj):
    ri = np.transpose(matli)
    rj = np.transpose(matlj)
    # sig_lj =  np.multiply(matlj, -1)
    # tmat = np.dot(matli,rj) + np.dot(matlj,ri)
    rtmat = np.dot(ri,matlj) + np.dot(rj,matli)
    # if np.array_equal(ri, inv(matli)) and np.array_equal(rj, inv(matlj)):
    return (np.count_nonzero(rtmat) == 0)

# Make all Adinkras?
def make_adinkras(k, legal_matrices):
    # """
    # Make all legal good lists of matrices of size k inside legal_matrices
    # """
    """ k is the color number of Adinkra, k=4 is a 4 color, 4 L Matrix Adinkra
        ie a tetrads
    """
    # main_list = [[] for i in range(len(legal_matrices))]
    # print(len(main_list))
    """ Preallocate lists """
    xtest_pack  = [None] * 12
    fourpack    = [None] * 4

    if k == 1:
        return [list(l) for l in legal_matrices]
    else:
        # adinkra_list = [None] * 36864
        adinkra_list    = []
        #print("Length lmats", len(legal_matrices))

        for i, mat in enumerate(legal_matrices):            # Find all matrix pairs for mat
            # good_mats = [m for m in legal_matrices if pairing(mat,m)]
            # test_mats = [ind[0] for ind in enumerate(legal_matrices) if pairing(mat, ind[1])]
            xtest_pack = [ind for ind in enumerate(legal_matrices) if pairing(mat, ind[1])]
            # main_list[i] = xtest_pack
            for val in xtest_pack:
                # main_list[val[0]] = [(i,mat)]
                fourpack    = [nmat for nmat in xtest_pack if pairing(val[1], nmat[1])]
                # print(len(fourpack))
                for xval in fourpack:
                    for lastx in [nmat for nmat in fourpack if pairing(xval[1], nmat[1])]:
                        # temp = [(i,mat), val, xval, lastx]
                        adinkra_list.append([(i,mat), val, xval, lastx])
        return adinkra_list

# ****************
def makeall_adinkras(k):

    n = k
    
    main_tetrad = make_adinkras(k, gen_product_matrices(k))
    print("\nFound " + str(len(main_tetrad)) + " tetrads \n")
    #for i in main_tetrad:
    #    print(i)

    return main_tetrad

# ******************************************************************************
# Calculate all determinants of L matrices in tetrad list.
def calc_det(main_tetrad):
    
    det_tetrad = []
    for tets in tqdm(main_tetrad):
        det_list = []
        mat_list = []
        for lmat in tets:
            det_list.append(np.linalg.det(lmat[1]))
            mat_list.append(lmat[0])
        det_tetrad.append([mat_list, det_list])
    return det_tetrad

# ******************************************************************************
# Calculate all traces of L matrices in tetrad list.
def calc_tra(main_tetrad):

    tra_tetrad = []
    for tets in tqdm(main_tetrad):
        tra_list = []
        mat_list = []
        for lmat in tets:
            tra_list.append(np.count_nonzero(np.diag(lmat[1])))
            mat_list.append(lmat[0])
        tra_tetrad.append([mat_list, tra_list])
    return tra_tetrad

# ******************************************************************************
# Count all signed elements of L matrices along rows.
def count_row(main_tetrad):

    row_tetrad = []
    for tets in tqdm(main_tetrad):
        row_list = []
        mat_list = []
        matsum = np.zeros((4, 4))
        for lmat in tets:
            matsum = np.add(matsum, lmat[1])
            mat_list.append(lmat[0])
        for row in matsum:
            row_list.append((row < 0).sum())
        row_tetrad.append([mat_list, row_list])
    return row_tetrad

# ******************************************************************************
# Count all signed elements of L matrices along columns.
def count_col(main_tetrad):

    col_tetrad = []
    for tets in tqdm(main_tetrad):
        col_list = []
        mat_list = []
        matsum = np.zeros((4, 4))
        for lmat in tets:
            matsum = np.add(matsum, lmat[1])
            mat_list.append(lmat[0])
        matsum = np.rot90(matsum,3)
        for row in matsum:
            col_list.append((row < 0).sum())
        col_tetrad.append([mat_list, col_list])
    return col_tetrad

# ******************************************************************************
# Export all Adinkra classes as quadrilaterals in sign-permutation space.
def draw_classes(class_matr):

    import pygame

# ******************************************************************************
# Find all patterns within the list of tetrads.
def pie_slicing(big_list_oftetrads):

    self_kein    = []
    kein_flip    = []

    for ind, itet in enumerate(big_list_oftetrads):
        # ivt = [n for n ]
        ivt = [np.transpose(xm[1]) for xm in itet]
        # for i in range(0, len(itet)):
        #     if np.array_equal(ivt[i], )
        for jnd, jtet in enumerate(big_list_oftetrads):

            if ind != jnd:
                if np.array_equal(ivt[0], jtet[0][1]) and np.array_equal(ivt[1], jtet[1][1]):
                    if np.array_equal(ivt[2], jtet[2][1]) and np.array_equal(ivt[3], jtet[3][1]):
                        print("Klein Flip found")
                        print("", ind, jnd)
                        demp = [ind, jnd]
                        demp.sort()
                        if demp not in kein_flip:
                            kein_flip.append(demp)
                        else:
                            print("Duplicate Klein")
                            pass
                        # kein_flip.append((ind, jnd))
            elif ind == jnd:
                if any(m for m in jtet if np.array_equal(ivt[0], m[1])) and any(m for m in jtet if np.array_equal(ivt[1], m[1])):
                    if any(m for m in jtet if np.array_equal(ivt[2], m[1])) and any(m for m in jtet if np.array_equal(ivt[3], m[1])):
                # if np.array_equal(ivt[0], jtet[0][1]) and np.array_equal(ivt[1], jtet[1][1]):
                    # if np.array_equal(ivt[2], jtet[2][1]) and np.array_equal(ivt[3], jtet[3][1]):
                        print("Self Kein Flip found")
                        print("", ind, jnd)
                        demp = [ind, jnd]
                        if demp not in self_kein:
                            self_kein.append(demp)
                        else:
                            print("Duplicate Self Klein")
                            pass

    print("")
    print("Length of Kein Flip list:", len(kein_flip))
    print("")
    print("Length of Self Kein Flip:", len(self_kein))

def findflip(iden):
    
    flipiden = str(iden)[:3]
    if int(iden[-4]) < 2:
        flipiden = flipiden + str(15 - int(iden[-4:-2])).zfill(2)
    else:
        if int(iden[-4]) == 2:
            flipiden = flipiden + str(10 + int(iden[-4:-2])).zfill(2)
        elif int(iden[-4]) == 3:
            flipiden = flipiden + str(-10 + int(iden[-4:-2])).zfill(2)
    if int(iden[-2]) < 2:
        flipiden = flipiden + str(15 - int(iden[-2:])).zfill(2)
    else:
        if int(iden[-2]) == 2:
            flipiden = flipiden + str(10 + int(iden[-2:])).zfill(2)
        elif int(iden[-2]) == 3:
            flipiden = flipiden + str(-10 + int(iden[-2:])).zfill(2)
    return flipiden

def flatten(lst):
    for elem in lst:
        if isinstance(elem, (list, tuple)):
            for nested in flatten(elem):
                yield nested
        else:
            yield elem

def nested(x, ys):
    return list(x.issuperset([nested]) for nested in flatten(ys))

def perm_parity(a,b):
    """Modified from
    http://code.activestate.com/recipes/578236-alternatve-generation-of-the-parity-or-sign-of-a-p/"""
    
    a = list(a)
    b = list(b)

    if sorted(a) != sorted(b): return 0
    inversions = 0
    while a:
        first = a.pop(0)
        inversions += b.index(first)
        b.remove(first)
    return -1 if inversions % 2 else 1

def loop_recursive(dim,n,q,s,paritycheck):
    if n < dim:
        for x in range(dim):
            q[n] = x
            loop_recursive(dim,n+1,q,s,paritycheck)
    else:
        s.append(perm_parity(q,paritycheck))
        
def LeviCivitaTensor(dim):
    qinit = np.zeros(dim)
    paritycheck = range(dim)
    flattened_tensor = []
    loop_recursive(dim,0,qinit,flattened_tensor,paritycheck)

    return np.reshape(flattened_tensor,[dim]*dim)

def LCvalue(dim, i, j, k, l):
    return LeviCivitaTensor(dim)[i-1][j-1][k-1][l-1]

def gadgetVV(allVtilde1, allVtilde2):
    value = 0
    LC = [[[[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0]],[[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  1],[ 0,  0, -1,  0]],[[ 0,  0,  0,  0],[ 0,  0,  0, -1],[ 0,  0,  0,  0],[ 0,  1,  0,  0]],[[ 0,  0,  0,  0],[ 0,  0,  1,  0],[ 0, -1,  0,  0],[ 0,  0,  0,  0]]], [[[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0, -1],[ 0,  0,  1,  0]],[[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0]],[[ 0,  0,  0,  1],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[-1,  0,  0,  0]],[[ 0,  0, -1,  0],[ 0,  0,  0,  0],[ 1,  0,  0,  0],[ 0,  0,  0,  0]]], [[[ 0,  0,  0,  0],[ 0,  0,  0,  1],[ 0,  0,  0,  0],[ 0, -1,  0,  0]],[[ 0,  0,  0, -1],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 1,  0,  0,  0]],[[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0]],[[ 0,  1,  0,  0],[-1,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0]]], [[[ 0,  0,  0,  0],[ 0,  0, -1,  0],[ 0,  1,  0,  0],[ 0,  0,  0,  0]],[[ 0,  0,  1,  0],[ 0,  0,  0,  0],[-1,  0,  0,  0],[ 0,  0,  0,  0]],[[ 0, -1,  0,  0],[ 1,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0]],[[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0]]]]
    for i in prange(4):
        for j in prange(4):
            for k in prange(4):
                for l in prange(4):
                    LCtemp = LC[i-1][j-1][k-1][l-1]
                    if  LCtemp == 0: continue
                    trace = np.einsum('ij,ji->', np.asmatrix(allVtilde1[i][j]), np.asmatrix(allVtilde2[k][l]))
                    value = value + (LCtemp * trace)
    return F('1/2')*F('1/48')*value

# ****************
# Run main()
start_time = time.time()
n = 4

main_tetrad = makeall_adinkras(n)
all_matrix = list(set([tuple((item[0], tuple(map(tuple, np.matrix.tolist(np.asmatrix(item[1])))))) for sublist in main_tetrad for item in sublist]))
Lmatr_dict = dict(all_matrix)

#for i in matr_dict:
#print(i)

#pie_slicing(main_tetrad)

#print main_tetrad

#exit()

sign_pmat = gen_signm(n)
uperm_mat = gen_permutations(n)

perm_dict = {tuple(map(tuple, matrix)):index for index,matrix in enumerate(uperm_mat)}
sign_dict = {tuple(map(tuple, matrix)):index for index,matrix in enumerate(sign_pmat)}

for k in Lmatr_dict.iterkeys():
    dict_tupl = Lmatr_dict[k]
    perm_temp = np.asmatrix(dict_tupl)
    perm_temp[perm_temp != 0] = 1
    sign_temp = np.asmatrix(np.dot(np.asmatrix(dict_tupl), np.linalg.inv(perm_temp)))
    perm_tupl = tuple(map(tuple, np.matrix.tolist(perm_temp)))
    sign_tupl = tuple(map(tuple, np.matrix.tolist(sign_temp)))
    Lmatr_dict[k] = (dict_tupl, perm_dict[perm_tupl], sign_dict[sign_tupl])

perm_dict_lookup = perm_dict
sign_dict_lookup = sign_dict

perm_dict = {index:tup for tup,index in perm_dict.items()}
sign_dict = {index:tup for tup,index in sign_dict.items()}

#print perm_dict, "\n"
#print sign_dict, "\n"
#print Lmatr_dict, "\n"
#print main_tetrad, "\n"

Rmatr_dict = dict()
for index,matrix in Lmatr_dict.items():
    matr,perm,sign = np.asmatrix(matrix[0]),matrix[1],matrix[2]
    trans = np.asmatrix(matr).transpose()
    tempinv = perm_dict_lookup[tuple(map(tuple, np.matrix.tolist(np.absolute(trans))))]
    for tup,invsign in sign_dict_lookup.items():
        if tuple(map(tuple, np.matrix.tolist(np.dot(np.asmatrix(tup), np.asmatrix(perm_dict[tempinv]))))) == tuple(map(tuple, np.matrix.tolist(trans))):
            tempsign = invsign
            break

    Rmatr = trans
    Rmatr_dict[index] = tuple((tuple(map(tuple, np.matrix.tolist(Rmatr))), tempinv, tempsign))

#print Lmatr_dict, "\n"
#print Rmatr_dict, "\n"
"""
f = open('Lmatr_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in Lmatr_dict.items():
    f.write("\n{} -> \n".format(key))
    f.write("{}\n".format(value))

f.close()
"""
"""
f = open('Rmatr_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in Rmatr_dict.items():
    f.write("\n{} -> \n".format(key))
    f.write("{}\n".format(value))

f.close()
"""
adinkra_dict = {index:tuple((adinkra[0][0], adinkra[1][0], adinkra[2][0], adinkra[3][0])) for index,adinkra in enumerate(main_tetrad)}
#print adinkra_dict
"""
f = open('adinkra_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in adinkra_dict.items():
    f.write("\n{} -> \n".format(key))
    for matr in value:
            f.write("{}\n".format(Lmatr_dict[matr]))

f.close()
"""
#for index,matrix in Lmatr_dict.items():
#    print np.dot(np.asmatrix(Rmatr_dict[index][0]), np.asmatrix(Lmatr_dict[index][0]))

print("# ********************************")
print("     ")
print("Generating L-R Dictionary")
print("     ")
"""
LiRj_dict = dict()
with tqdm(total=len(adinkra_dict.items())) as pbar_adinkra:
    for index,adink in adinkra_dict.items():
        if index not in LiRj_dict.iterkeys(): 
            LiRj_dict[index] = []
        for i in adink:
            templist = []
            for j in adink:
                #print np.asmatrix(Lmatr_dict[i][0]), np.asmatrix(Rmatr_dict[j][0])
                LiRj = np.dot(np.asmatrix(Lmatr_dict[i][0]), np.asmatrix(Rmatr_dict[j][0]))
                tempperm = perm_dict_lookup[tuple(map(tuple, np.matrix.tolist(np.absolute(LiRj))))]
                for tup,invsign in sign_dict_lookup.items():
                    if tuple(map(tuple, np.matrix.tolist(np.dot(np.asmatrix(tup), np.asmatrix(perm_dict[tempperm]))))) == tuple(map(tuple, np.matrix.tolist(LiRj))):
                            tempsign = invsign
                            break
                templist.append(tuple((tuple(map(tuple, np.matrix.tolist(LiRj))), tempperm, tempsign)))
            LiRj_dict[index].append(tuple(templist))
        pbar_adinkra.update(1)

print LiRj_dict

f = open('LiRj_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in LiRj_dict.items():
    f.write("\n{} -> \n".format(key))
    for matrlist in value:
        for matr in matrlist:
            f.write("{}\n".format(matr))

f.close()
"""
"""
num_lines = sum(1 for line in open('LiRj_export.txt'))
inputfile = open("LiRj_export.txt","r")

LiRj_dict = dict()
adink = 0
matrlist = []
fulllist = []

with tqdm(total=626688) as pbar_lines:
    for line in inputfile.readlines():
        if len(matrlist) == 4:
            fulllist.append(tuple(matrlist))
            matrlist = []
        terms = line.split()
        if not line.strip():
            continue
        if terms[1] == "->":
            LiRj_dict[adink] = tuple(fulllist)
            matrlist = []
            fulllist = []
            adink = int(terms[0])
        else:
            newline = tuple(eval(line))
            matrlist.append(tuple((tuple(map(tuple, newline[0])), int(newline[1]), int(newline[2]))))
        pbar_lines.update(1)
    newline = tuple(eval(line))
    fulllist.append(tuple(matrlist))
    LiRj_dict[adink] = tuple(fulllist)

with tqdm(total=len(adinkra_dict.items())) as pbar_adinkra:
    for index,adink in adinkra_dict.items():
        print LiRj_dict[index]
"""
print("     ")
print("# ********************************")
print("     ")
print("Generating R-L Dictionary")
print("     ")
"""
RiLj_dict = dict()
with tqdm(total=len(adinkra_dict.items())) as pbar_adinkra:
    for index,adink in adinkra_dict.items():
        if index not in RiLj_dict.iterkeys(): 
            RiLj_dict[index] = []
        for i in adink:
            templist = []
            for j in adink:
                #print np.asmatrix(Lmatr_dict[i][0]), np.asmatrix(Rmatr_dict[j][0])
                RiLj = np.dot(np.asmatrix(Rmatr_dict[i][0]), np.asmatrix(Lmatr_dict[j][0]))
                tempperm = perm_dict_lookup[tuple(map(tuple, np.matrix.tolist(np.absolute(RiLj))))]
                for tup,invsign in sign_dict_lookup.items():
                    if tuple(map(tuple, np.matrix.tolist(np.dot(np.asmatrix(tup), np.asmatrix(perm_dict[tempperm]))))) == tuple(map(tuple, np.matrix.tolist(RiLj))):
                            tempsign = invsign
                            break
                templist.append(tuple((tuple(map(tuple, np.matrix.tolist(RiLj))), tempperm, tempsign)))
            RiLj_dict[index].append(tuple(templist))
        pbar_adinkra.update(1)

print RiLj_dict

f = open('RiLj_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in RiLj_dict.items():
    f.write("\n{} -> \n".format(key))
    for matrlist in value:
        for matr in matrlist:
            f.write("{}\n".format(matr))

f.close()
"""
"""
num_lines = sum(1 for line in open('RiLj_export.txt'))
inputfile = open("RiLj_export.txt","r")

RiLj_dict = dict()
adink = 0
matrlist = []
fulllist = []

with tqdm(total=626688) as pbar_lines:
    for line in inputfile.readlines():
        if len(matrlist) == 4:
            fulllist.append(tuple(matrlist))
            matrlist = []
        terms = line.split()
        if not line.strip():
            continue
        if terms[1] == "->":
            RiLj_dict[adink] = tuple(fulllist)
            matrlist = []
            fulllist = []
            adink = int(terms[0])
        else:
            newline = tuple(eval(line))
            matrlist.append(tuple((tuple(map(tuple, newline[0])), int(newline[1]), int(newline[2]))))
        pbar_lines.update(1)
    newline = tuple(eval(line))
    fulllist.append(tuple(matrlist))
    RiLj_dict[adink] = tuple(fulllist)

with tqdm(total=len(adinkra_dict.items())) as pbar_adinkra:
    for index,adink in adinkra_dict.items():
        print RiLj_dict[index]
"""
print("     ")
print("# ********************************")
print("     ")
print("Generating V Dictionary")
print("     ")
"""
V_dict = dict()
with tqdm(total=len(adinkra_dict.items())) as pbar_adinkra:
    for index,adink in adinkra_dict.items():
        if index not in V_dict.iterkeys(): 
            V_dict[index] = []
        for i in adink:
            templist = []
            for j in adink:
                part1 = np.dot(np.asmatrix(Lmatr_dict[i][0]), np.asmatrix(Rmatr_dict[j][0]))
                part2 = np.dot(np.asmatrix(Lmatr_dict[j][0]), np.asmatrix(Rmatr_dict[i][0]))
                V = -(part1 - part2)/2
                templist.append(tuple((tuple(map(tuple, np.matrix.tolist(V))))))
            V_dict[index].append(tuple(templist))
        pbar_adinkra.update(1)

f = open('V_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in V_dict.items():
    f.write("\n{} -> \n".format(key))
    for matrlist in value:
        for matr in matrlist:
            f.write("{}\n".format(matr))

f.close()
"""
num_lines = sum(1 for line in open('V_export.txt'))
inputfile = open("V_export.txt","r")

V_dict = dict()
adink = 0
matrlist = []
fulllist = []

with tqdm(total=626688) as pbar_lines:
    for line in inputfile.readlines():
        if len(matrlist) == 4:
            fulllist.append(tuple(matrlist))
            matrlist = []
        terms = line.split()
        if not line.strip():
            continue
        if terms[1] == "->":
            V_dict[adink] = tuple(fulllist)
            matrlist = []
            fulllist = []
            adink = int(terms[0])
        else:
            newline = tuple(eval(line))
            matrlist.append(tuple((tuple(map(tuple, newline)))))
        pbar_lines.update(1)
    newline = tuple(eval(line))
    fulllist.append(tuple(matrlist))
    V_dict[adink] = tuple(fulllist)
#print V_dict

print("     ")
print("# ********************************")
print("     ")
print("Generating V-tilde Dictionary")
print("     ")
"""
Vtilde_dict = dict()
with tqdm(total=len(adinkra_dict.items())) as pbar_adinkra:
    for index,adink in adinkra_dict.items():
        if index not in Vtilde_dict.iterkeys():
            Vtilde_dict[index] = []
        for i in adink:
            templist = []
            for j in adink:
                part1 = np.dot(np.asmatrix(Rmatr_dict[i][0]), np.asmatrix(Lmatr_dict[j][0]))
                part2 = np.dot(np.asmatrix(Rmatr_dict[j][0]), np.asmatrix(Lmatr_dict[i][0]))
                Vtilde = -(part1 - part2)/2
                templist.append(tuple((tuple(map(tuple, np.matrix.tolist(Vtilde))))))
            Vtilde_dict[index].append(tuple(templist))
        pbar_adinkra.update(1)

f = open('Vtilde_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in Vtilde_dict.items():
    f.write("\n{} -> \n".format(key))
    for matrlist in value:
        for matr in matrlist:
            f.write("{}\n".format(matr))

f.close()
"""

num_lines = sum(1 for line in open('Vtilde_export.txt'))
inputfile = open("Vtilde_export.txt","r")

Vtilde_dict = dict()
adink = 0
matrlist = []
fulllist = []

with tqdm(total=626688) as pbar_lines:
    for line in inputfile.readlines():
        if len(matrlist) == 4:
            fulllist.append(tuple(matrlist))
            matrlist = []
        terms = line.split()
        if not line.strip():
            continue
        if terms[1] == "->":
            Vtilde_dict[adink] = tuple(fulllist)
            matrlist = []
            fulllist = []
            adink = int(terms[0])
        else:
            newline = tuple(eval(line))
            matrlist.append(tuple((tuple(map(tuple, newline)))))
        pbar_lines.update(1)
    newline = tuple(eval(line))
    fulllist.append(tuple(matrlist))
    Vtilde_dict[adink] = tuple(fulllist)
#print Vtilde_dict
"""
pool = Pool(processes=300)
while True:
    adink1 = input("adink1: ")
    adink2 = input("adink2: ")
    result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink2],))
    print result.get()
exit()
"""

pool = Pool(processes=300)

adinklist = [22730]
#adinklist = [22709, 21749, 26837, 30933]

gadgetvalues = []

with tqdm(total=len(adinklist)) as pbar_gadget1:
    for adink1 in adinklist:
        result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink1],))
        gadgetvalues.append([adink1, result.get()])
        pbar_gadget1.update(1)

print gadgetvalues

exit()

nego = 0
negt = 0
post = 0
poso = 0

pool = Pool(processes=300)
lower = input("lower: ")
upper = input("upper: ")
with tqdm(total=upper-lower+1) as pbar_gadget1:
    for adink1 in prange(int(lower), int(upper)+1):
        #adink1 = input("adink1: ")
        gadgetvalues = []
        with tqdm(total=36863-adink1) as pbar_gadget2:
            for adink2 in prange(adink1+1, 36864):
                result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink2],))
                gadgetvalues.append(result.get())
                pbar_gadget2.update(1)
        countlist = np.negative(np.array(gadgetvalues).astype(np.float))
        c = collections.Counter(countlist)
        print c
        nego += c[-1.0]
        negt += c[-0.33333333333333331]
        post += c[0.33333333333333331]
        poso += c[1.0]
        print nego, negt, post, poso
        pbar_gadget1.update(1)

exit()

pool = Pool(processes=300)
lower = 0
upper = 95

gadgetgraph = []
with tqdm(total=upper-lower) as pbar_gadget1:
    for adink1 in prange(int(lower), int(upper)):
        gadgetvalues = []
        with tqdm(total=95-adink1) as pbar_gadget2:
            for adink2 in prange(adink1+1, 96):
                #print adink1, adink2
                result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink2],))
                gadgetvalues.append(result.get())
                pbar_gadget2.update(1)
        paddedvalues = np.pad(gadgetvalues,(adink1+1,0), 'constant')
        #print paddedvalues, len(paddedvalues)
        pbar_gadget1.update(1)

        gadgetgraph.append([paddedvalues])

listofzeros = [0] * 96
gadgetgraph.append([listofzeros])

tempcount1 = [item for sublist in gadgetgraph for item in sublist]
print tempcount1[0]
tempcount = [item for sublist in tempcount1 for item in sublist]
print len(tempcount)
gadgetsmall = np.negative(np.array(tempcount).astype(np.float))
print collections.Counter(gadgetsmall)

A = np.reshape(gadgetsmall, newshape=(96,96))
B = A + A.T - np.diag(np.diag(A))

plt.figure()
plt.imshow(B,interpolation='none')#, vmin=0, vmax=1, aspect='equal')
plt.tight_layout()

ax = plt.gca()
ax = plt.gca()

# Major ticks
ax.set_xticks(np.arange(0, 96, 1))
ax.set_yticks(np.arange(0, 96, 1))

# Labels for major ticks
ax.set_xticklabels(np.arange(1, 97, 1))
ax.set_yticklabels(np.arange(1, 97, 1))

# Minor ticks
ax.set_xticks(np.arange(-.5, 96, 1), minor=True)
ax.set_yticks(np.arange(-.5, 96, 1), minor=True)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax.yaxis.tick_left()
ax.yaxis.set_label_position('left')

plt.setp(ax.get_xticklabels(), fontsize=3)
plt.setp(ax.get_yticklabels(), fontsize=3)

# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=.5)

#plt.show()
plt.savefig('gadget_small.png', format='png', dpi=1000, transparent=True)

exit()

pool = Pool(processes=300)
lower = input("lower: ")
upper = input("upper: ")
with tqdm(total=upper-lower+1) as pbar_gadget1:
    for adink1 in prange(int(lower), int(upper)+1):
        #adink1 = input("adink1: ")
        gadgetvalues = []
        with tqdm(total=36863-adink1) as pbar_gadget2:
            for adink2 in prange(adink1+1, 36864):
                result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink2],))
                gadgetvalues.append([''.join('(' + str(adink1) + ', ' + str(adink2) + ')'), result.get()])
                pbar_gadget2.update(1)
        f = open('gadgetvalues_export/values/'+str(adink1)+'.txt', 'w')
        first = True
        for item in gadgetvalues:
            if first:
                f.write("{} -> {} ".format(item[0], item[1]))
                first = False
            else:
                f.write("\n{} -> {} ".format(item[0], item[1]))
            #f.write("{}\n".format(item[1]))
        f.close()
        pbar_gadget1.update(1)
exit()

print("     ")
print("# ********************************")
print("     ")
print("Calculating New Gadget from V-tilde Matrices")
print("     ")
pool = Pool(processes=300)
with tqdm(total=243) as pbar_group: #1536
    for group in prange(1293, 1536): #256,320 --> 1536
        #320, 564, 807, 1050, 1293, 1536
        gadgetvalues = []
        with tqdm(total=24) as pbar_gadget1:
            for adink1 in prange(group*24, (group+1)*24):
                with tqdm(total=36863-adink1) as pbar_gadget2:
                    for adink2 in prange(adink1+1, 36864):
                        result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink2],))
                        gadgetvalues.append([''.join('(' + str(adink1) + ', ' + str(adink2) + ')'), result.get()])
                        pbar_gadget2.update(1)
                pbar_gadget1.update(1)
        f = open('gadgetvalues_export.txt', 'a+')
        for item in gadgetvalues:
            f.write("\n{} -> \n".format(item[0]))
            f.write("{}\n".format(item[1]))
        f.close()
        pbar_group.update(1)
"""
pool = Pool(processes=300)
with tqdm(total=64) as pbar_group: #1536
    for group in prange(256,320):
        gadgetvalues = dict()
        with tqdm(total=24) as pbar_gadget1:
            for adink1 in prange(group*24, (group+1)*24):
                with tqdm(total=36863-adink1) as pbar_gadget2:
                    for adink2 in prange(adink1+1, 36864):
                        result = pool.apply_async(gadgetVV, (Vtilde_dict[adink1], Vtilde_dict[adink2],))
                        gadgetvalues[''.join('(' + str(adink1) + ', ' + str(adink2) + ')')] = result.get()
                        pbar_gadget2.update(1)
                pbar_gadget1.update(1)
        f = open('gadgetvalues_export.txt', 'a+')
        for key, value in gadgetvalues.items():
            f.write("\n{} -> \n".format(key))
            f.write("{}\n".format(value))
        f.close()
        pbar_group.update(1)
"""
"""
pool = Pool(processes=8)
LC = LeviCivitaTensor(4)
gadgetvalues = dict()
with tqdm(total=36864) as pbar_gadget1:
    for adink1 in prange(36864):
        allVtilde1 = Vtilde_dict[adink1]
        with tqdm(total=36863-adink1) as pbar_gadget2:
            for adink2 in prange(adink1+1, 36864):
                allVtilde2 = Vtilde_dict[adink2]
                result = pool.apply_async(gadgetVV, (LC, allVtilde1, allVtilde2,))
                gadgetvalues[''.join('(' + str(adink1) + ', ' + str(adink2) + ')')] = result.get()
                pbar_gadget2.update(1)
        pbar_gadget1.update(1)
f = open('gadgetvalues_export.txt', 'a+')
for key, value in gadgetvalues.items():
    f.write("\n{} -> \n".format(key))
    f.write("{}\n".format(value))
f.close()
"""
"""
LC = LeviCivitaTensor(4)
gadgetvalues = dict()
with tqdm(total=36864) as pbar_gadget1:
    for adink1 in prange(36864):
        allVtilde1 = Vtilde_dict[adink1]
        with tqdm(total=36863-adink1) as pbar_gadget2:
            for adink2 in prange(adink1+1, 36864):
                allVtilde2 = Vtilde_dict[adink2]
                value = 0
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            for l in range(4):
                                LCtemp = LC[i-1][j-1][k-1][l-1]
                                if  LCtemp == 0: continue
                                trace = np.einsum('ij,ji->', np.asmatrix(allVtilde1[i][j]), np.asmatrix(allVtilde2[k][l]))
                                value = value + (LCtemp * trace)
                gadgetvalues[str((adink1, adink2))] = F('1/2')*F('1/48')*value
                pbar_gadget2.update(1)
        pbar_gadget1.update(1)
"""

"""
print("     ")
print("# ********************************")
print("     ")
print("Labelling and Categorizing V Matrices")
print("     ")

inputfile = open("Vpart_dict_export.txt","r")

Vpart_lookup = dict()
num = "00"

for line in inputfile.readlines():
    terms = line.split()
    if len(terms) == 0: continue
    if terms[1] == "->":
        num = terms[0].zfill(2)
    else:
        newline = tuple(eval(line))
        Vpart_lookup[num] = tuple((tuple(map(tuple, newline[0])), str(newline[1]), str(newline[2])))

inputfile.close()

#print Vpart_lookup

temp_Vpart_lookup = dict()
for key, value in Vpart_lookup.items():
    temp_Vpart_lookup[value[0]] = key

#print temp_Vpart_lookup

Vclass_dict_lookup = dict()
for key, value in V_dict.items():
    tempset = []
    for matrlist in value:
        for matr in matrlist:
            tempset.append(temp_Vpart_lookup[matr])
    Vclass_dict_lookup[key] = ''.join(tempset)

Vclass_dict = {}
for key, value in Vclass_dict_lookup.items():
    if value not in Vclass_dict.iterkeys():
        Vclass_dict[value] = []
    Vclass_dict[value].append(key)

f = open('Vclass_dict_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in Vclass_dict.items():
    #split = re.findall('.',key)
    #keyname = ''.join(split[2] + split[3] + split[5] + split[7] + split[13] + split[15] + split[23])
    f.write("\n{} -> \n".format(key))
    for adink in value:
            f.write("{}\n".format(adink))

f.close()

f = open('class_visual_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in Vclass_dict.items():
    keyvisual = re.findall('..',key)
    keyvisual = list(''.join(t + ' ') for t in keyvisual)
    keyvisual = list(''.join(l + '\n' * (n % 4 == 3) for n, l in enumerate(keyvisual)))
    keyvisual[len(keyvisual)-1] = " -> "
    f.write("\n{}\n".format(''.join(keyvisual)))
    for adink in value:
            f.write("{}\n".format(adink))

f.close()

f = open('class_export.txt', 'w')
#print >> f, 'Filename:', filename  # or f.write('...\n')

for key, value in Vclass_dict.items():
    split = re.findall('.',key)
    keyname = ''.join(split[2] + split[3] + split[5] + split[7] + split[13] + split[15] + split[23])
    f.write("\n{} -> \n".format(keyname))
    for adink in value:
        listmatr = list(Lmatr_dict[i][0] for i in adinkra_dict[adink])
        f.write("{}\n".format(tuple(listmatr)))

f.close()
"""
#re.findall('..','classname')

print("\n-- Execution time --")
print("---- %s seconds ----" % (time.time() - start_time))
