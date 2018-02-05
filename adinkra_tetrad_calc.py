# ******************************************************************************
# Name:    Caculate all BC4-based Adinkras
# Author:  Vadim Korotkikh
# Email:   va.korotki@gmail.com
# Date:    December 2016
# Version: 1.3A
#
# Description: The code calculates all all unique 36,864 ordered BC4-based
# adinkras with four colors, four open-nodes and four closed nodes and then
# calculates the Vij holoraumy matrices and the Gadget values
#
# ******************************************************************************


# ******************************************************************************
# Library Imports
import math
import sys
import numpy as np
import numpy.matlib
import itertools
from numpy import array
from numpy.linalg import inv
import time
from tqdm import *
 
# ******************************************************************************
# Function Imports
import alpha_beta_4x4
import vij_holoraumy_calc

# ******************************************************************************
# Main() function.
def main():
    print("# ***********************************************************************")
    print("#    ")
    print("# Name:    Caculate all BC4-based Adinkras")
    print("# Author:  Vadim Korotkikh    ")
    print("# Email:   va.korotki@gmail.com")
    print("# Date:    December 2016        ")
    print("# Version: 1.3A                ")
    print("#                            ")
    print("# Description: Calculates all unique 36,864 ordered BC4-based adinkras")
    print("# with four colors, four open-nodes and four closed nodes.             ")
    print("#    ")
    print("# ***********************************************************************")
    print("        ")
    calculate_wisdom(4)

# ******************************************************************************
# Calculate all possible Adinkra (L matrix tetrads) in BC4 space
def calculate_wisdom(n):

    """ Main tetrad list """
    numonly_tetrads = []
    tetrad_reftup = []

    res = []
    calc_check = []
    pcal_check = []

    invs_check = []
    trnp_check = []

    sign_pmat = gen_signm(n)
    uperm_mat = gen_dpermut_mat()

    duplik_one       = []
    duplik_extras    = []

    for x in sign_pmat:
        t1 = np.asmatrix(x)
        for y in uperm_mat:
            t2 = np.asmatrix(y)
            dprod = np.dot(t1,t2)
            dprod2 = np.matmul(t1,t2)
            if np.array_equal(dprod, dprod2):
                res.append(dprod)
                if duplik_one:
                    if [mx for mx in duplik_one if np.array_equal(dprod, mx)]:
                        duplik_extras.append(dprod)
                    else:
                        duplik_one.append(dprod)
                else:
                    duplik_one.append(dprod)

                sigflip = np.multiply(dprod, -1)
                trnpmat = np.transpose(dprod)
                if calc_check:
                    if [mt for mt in calc_check if np.array_equal(mt, sigflip)]:
                        pcal_check.append(dprod)
                    else:
                        calc_check.append(dprod)
                else:
                    calc_check.append(dprod)

    for i, im in enumerate(res):
        invsmat = inv(im)
        trnpmat = np.transpose(im)
        # print(im)
        for j, jm in enumerate(res):
            if i != j:
                temp = [i, j]
                if np.array_equal(invsmat, jm):
                    # print("INVERSE")
                    # print("i, j:", i, j)
                    # temp = [i, j]
                    temp.sort()
                    if temp not in invs_check:
                        invs_check.append(temp)
                        # print("INVERSE")
                        # print("i j", i, j)
                    else:
                        pass
                if np.array_equal(trnpmat, jm):
                    # temp = [i, j]
                    temp.sort()
                    if temp not in trnp_check:
                        trnp_check.append(temp)
                        # print("TRANSPOSE")
                        # print("i j", i, j)
                    else:
                        pass

    print("Finishing building 384 matrices")
    print("Checking for duplicates in 384")
    print("Number of unique matrices:", len(duplik_one))
    print("Number of duplicate matrices:", len(duplik_extras))
    print("Sign flips removed")
    print("Number of sign-flipped matrices:", len(calc_check))

    print("Inverse check")
    print("Number of inverse duplicate matrices:", len(invs_check))

    print("Transpose check")
    print("Number of Transpose duplicate matrices:", len(trnp_check))
    test_list = []
    for xmat in invs_check:
        if [mat for mat in trnp_check if np.array_equal(mat, xmat)]:
            test_list.append(xmat)
    print("Checking if Inverse and Transpose sets are the same")
    print("Number of Inverse - Transpose matches:", len(test_list))

    """ Creating the 2 * I, 4x4 identity matrix """
    idt_mat = 2 * np.matlib.identity(n, dtype=int)

    """ Start finding Li - Lj pairs that satisfy 2.4b, 2.4c and 2.4d """
    print("")
    print("---Finding Matrix Pairs---\n")

    for i, li in enumerate(tqdm(res)):
        tetrad_reftup.append((i, li))
        # Hold the Li and Lj matching matrices tuples
        # temp_tup    = []
        # Holds only the Lj matrices that match Li
        temp_m        = []
        """ Testing temps """
        # Temporary lists for error checking/debugging
        # temp_tst    = []

        ###print("Finding pairs for Li: ", i)

        for j, lj in enumerate(res):
            # sigflip_lj = np.multiply(lj, -1)
            ri = np.transpose(li)
            rj = np.transpose(lj)
            if i == j:
                if np.array_equal(ri,rj):
                    # tmat = np.dot(li,ri) + np.dot(li,ri)
                    tmat  = 2 * np.dot(li,ri)
                    rtmat = 2 * np.dot(ri, li)
                    #if np.array_equal(tmat, idt_mat) and np.array_equal(rtmat, idt_mat):
                        # print("EQ satisfied\n", tmat, idt_mat)
                        ###print("EQ 2.4a satisfied for I = J ",i,j)
                    #else:
                        #pass
                        # print("EQ 2.4a failed", i, j)
            elif i != j:
                tmat = np.dot(li,rj) + np.dot(lj,ri)
                rtmat = np.dot(ri,lj) + np.dot(rj,li)
                if np.count_nonzero(rtmat) == 0 and np.count_nonzero(tmat) == 0:
                # if np.count_nonzero(rtmat) == 0:
                    if np.array_equal(ri, inv(li)) and np.array_equal(rj, inv(lj)):
                        ###print("EQ 2.4d satisfied", i, j)
                        # packing away all the 12 matching matrices
                        temp_m.append([j,lj])
                        # print("Matching I J pair: ",i,j)
                        # temp_tup.append((li,lj))
                    else:
                        pass
                        # print("EQ 2.4d failed", i ,j)
                else:
                    """ Making sure the outputs make sense """
                    pass
                    # temp_tst.append([j,lj])
                    # print("EQ 2.4b, 2.4c failed", i, j)

                # """ Check whether the jth, lj matrix is a sign flip of li """
                # if np.array_equal(li,sigflip_lj):
                #     temp_l = [(i,li),(j,lj)]
                #     if sorted(temp_l,key=lambda item: item[0]) not in negs_filter:
                #         negs_filter.append(temp_l)

        temp_i_tetrad = build_four_pillars([i, li], temp_m)
        new_tets = [x for x in temp_i_tetrad if x not in numonly_tetrads]
        numonly_tetrads.extend(new_tets)

        ###print("Break point for new i:", i)
        ###print("Number of pair matrices:",len(temp_m))
        ###print("Length of numonly_tetrads list:", len(numonly_tetrads))
        # print("Number of sign-reduced matrices:", len(temp_tst))
        ###print("     <<>>    \n")
        temp = []
        temp_m = []
        # temp_tst = []
    
    main_tetrad = []
    # A tetrad is a collection of 4 matrices that all satify the pair conditions.
    print("        ")
    print("# ********************************")
    print("        ")
    print("Building tetrad matrix list")
    print("        ")
    #print("Printing tetrads")
    for tets in numonly_tetrads:
        new_tet = []
        for mat_num in tets:
            if tetrad_reftup[mat_num][0] == mat_num:
                temp_tup = tetrad_reftup[mat_num]
                new_tet.append(temp_tup)
        # print(new_tet)
        main_tetrad.append(new_tet)
    #print main_tetrad
    #print("main_tetrad:", main_tetrad[0])
    #print(main_tetrad[1])
    # mtetrad_size = sys.getsizeof(main_tetrad)
    # print("Size of main_tetrads list: bytes / kilobytes:", mtetrad_size, mtetrad_size/1024)
    print("Total number of unique tetrad permutations:", len(main_tetrad))

    all_matrix = list(set([tuple((item[0], tuple(map(tuple, np.matrix.tolist(item[1]))))) for sublist in main_tetrad for item in sublist]))
    matr_dict = dict(all_matrix)

    #for key, value in matr_dict.items():
    #    print("{} -> {}".format(key, value))
    
    # pie_slicing(main_tetrad)
    # vij_holoraumy_calc.calculate_vij_matrices(main_tetrad)
    
    print("     ")
    print("# ********************************")
    print("     ")
    print("Calculating determinants")
    print("     ")
    det_tetrad = calc_det(main_tetrad)
    #print det_tetrad

    print("     ")
    print("# ********************************")
    print("     ")
    print("Calculating traces")
    print("     ")
    tra_tetrad = calc_tra(main_tetrad)
    #print tra_tetrad

    print("     ")
    print("# ********************************")
    print("     ")
    print("Counting signs along row")
    print("     ")
    row_tetrad = count_row(main_tetrad)
    #print row_tetrad

    print("     ")
    print("# ********************************")
    print("     ")
    print("Counting signs along columns")
    print("     ")
    col_tetrad = count_col(main_tetrad)
    #print col_tetrad

    print("     ")
    print("# ********************************")
    print("     ")
    print("Tally Adinkra characteristics")
    print("     ")
    print("# ********************************")
    print("     ")

    from collections import Counter   
 
    only_det = [tuple(row[1]) for row in det_tetrad]
    print("Det Tally")
    print("     ")
    det_tally = Counter(only_det)
    for key, value in det_tally.items():
        print("{} -> {}".format(key, value))
    print("     ")

    only_tra = [tuple(row[1]) for row in tra_tetrad]
    print("Tra Tally")
    print("     ")
    tra_tally = Counter(only_tra)
    for key, value in tra_tally.items():
        print("{} -> {}".format(key, value))
    print("     ")

    only_row = [tuple(row[1]) for row in row_tetrad]
    print("Row Tally")
    print("     ")
    row_tally = Counter(only_row)
    for key, value in row_tally.items():
        print("{} -> {}".format(key, value))
    print("     ")

    only_col = [tuple(row[1]) for row in col_tetrad]
    print("Col Tally")
    print("     ")
    col_tally = Counter(only_col)
    for key, value in col_tally.items():
        print("{} -> {}".format(key, value))
    print("     ")

    det_dict = dict(zip([tuple(row[0]) for row in det_tetrad], only_det))
    #print det_dict

    tra_dict = dict(zip([tuple(row[0]) for row in tra_tetrad], only_tra))
    #print tra_dict

    row_dict = dict(zip([tuple(row[0]) for row in row_tetrad], only_row))
    #print row_dict

    col_dict = dict(zip([tuple(row[0]) for row in col_tetrad], only_col))
    #print col_dict

    dict_list = [det_dict, tra_dict, row_dict, col_dict]
    all_dict = {}
    for k in det_dict.iterkeys():
        all_dict[k] = tuple(d[k] for d in dict_list)

    #print all_dict
    #print len(all_dict)

    print("# ********************************")
    print("     ")
    print("Building Adinkra classes")
    print("     ")

    id1 = {(1, 1, 1, 1):"0", (-1, -1, -1, -1):"1"}
    id2 = {(0, 0, 0, 4):"01", (0, 0, 4, 0):"02", (0, 0, 2, 2):"03", (0, 4, 0, 0):"04", (0, 2, 0, 2):"05", (0, 2, 2, 0):"06", (4, 0, 0, 0):"08", (2, 0, 0, 2):"09", (2, 0, 2, 0):"10", (2, 2, 0, 0):"12", (1, 1, 1, 1):"15"}
    id3 = {(3, 3, 3, 3):"00", (3, 3, 3, 1):"01", (3, 3, 1, 3):"02", (3, 3, 1, 1):"03", (3, 1, 3, 3):"04", (3, 1, 3, 1):"05", (3, 1, 1, 3):"06", (3, 1, 1, 1):"07", (1, 3, 3, 3):"08", (1, 3, 3, 1):"09", (1, 3, 1, 3):"10", (1, 3, 1, 1):"11", (1, 1, 3, 3):"12", (1, 1, 3, 1):"13", (1, 1, 1, 3):"14", (1, 1, 1, 1):"15", (0, 2, 2, 2):"20", (2, 0, 2, 2):"21", (2, 2, 0, 2):"22", (2, 2, 2, 0):"23", (4, 2, 2, 2):"30", (2, 4, 2, 2):"31", (2, 2, 4, 2):"32", (2, 2, 2, 4):"33"}
    id4 = {(3, 3, 3, 3):"00", (3, 3, 3, 1):"01", (3, 3, 1, 3):"02", (3, 3, 1, 1):"03", (3, 1, 3, 3):"04", (3, 1, 3, 1):"05", (3, 1, 1, 3):"06", (3, 1, 1, 1):"07", (1, 3, 3, 3):"08", (1, 3, 3, 1):"09", (1, 3, 1, 3):"10", (1, 3, 1, 1):"11", (1, 1, 3, 3):"12", (1, 1, 3, 1):"13", (1, 1, 1, 3):"14", (1, 1, 1, 1):"15", (0, 2, 2, 2):"20", (2, 0, 2, 2):"21", (2, 2, 0, 2):"22", (2, 2, 2, 0):"23", (4, 2, 2, 2):"30", (2, 4, 2, 2):"31", (2, 2, 4, 2):"32", (2, 2, 2, 4):"33"}

    class_dict_rev = {}
    for k in all_dict.iterkeys():
        class_dict_rev[k] = ''.join(tuple([id1[all_dict[k][0]], id2[all_dict[k][1]], id3[all_dict[k][2]], id4[all_dict[k][3]]]))
    #print class_dict, len(class_dict)
    
    class_list = []
    for k in class_dict_rev.iterkeys():
        class_list.append(class_dict_rev[k])
    class_tally = Counter(class_list)
    for key, value in class_tally.items():
        print("{} -> {}".format(key, value))
    
    print("     ")
    print("There are {} distinct classes".format(len(class_tally)))
    print("     ")
    
    print("     ")
    print("# ********************************")
    print("     ")
    print("Build categorized L-matrix reference list")
    print("     ")

    perm_dict = {tuple(map(tuple, matrix)):index for index,matrix in enumerate(uperm_mat)}
    sign_dict = {tuple(map(tuple, matrix)):index for index,matrix in enumerate(sign_pmat)}
    
    for k in matr_dict.iterkeys():
        dict_tupl = matr_dict[k]
        perm_temp = np.asmatrix(dict_tupl)
        perm_temp[perm_temp != 0] = 1
        sign_temp = np.asmatrix(np.dot(np.asmatrix(dict_tupl), np.linalg.inv(perm_temp)))
        perm_tupl = tuple(map(tuple, np.matrix.tolist(perm_temp)))
        sign_tupl = tuple(map(tuple, np.matrix.tolist(sign_temp)))
        matr_dict[k] = (dict_tupl, perm_dict[perm_tupl], sign_dict[sign_tupl])
    
    #print matr_dict 
    #print perm_dict
    #print sign_dict
    #print class_dict_rev

    class_temp = []
    class_temp2 = []
    for k in class_dict_rev.iterkeys():
        matr_list = []
        for mat in k:
            matr_list.append(tuple([mat, matr_dict[mat][1], matr_dict[mat][2]]))
        class_temp.append([class_dict_rev[k], tuple(matr_list)])
        class_temp2.append([class_dict_rev[k], tuple(sorted(matr_list, key=lambda k: (k[2], k[1])))])

    #print class_temp, len(class_temp)

    class_dict = {}
    for elem in class_temp:
        if elem[0] in class_dict:
            class_dict[elem[0]].append(elem[1])
            class_dict[elem[0]] = sorted(class_dict[elem[0]], key=lambda k: (k[0][2], k[0][1]))
        else:
            class_dict[elem[0]] = [elem[1]]

    class_matr = {}
    for elem in class_temp2:
        if elem[0] in class_matr:
            class_matr[elem[0]].append(elem[1])
            class_matr[elem[0]] = sorted(class_matr[elem[0]], key=lambda k: (k[0][2], k[0][1]))
        else:
            class_matr[elem[0]] = [elem[1]]

    #f = open('class_dict.txt', 'w')
    #print >> f, 'Filename:', filename  # or f.write('...\n')
    #f.close()

    for key, value in class_matr.items():
        print("\n{} -> \n".format(key))
        matr_tally = Counter(value)
        for key in sorted(matr_tally.iterkeys(), key=lambda k: (k[0][2], k[0][1])):
            print("    {} -> {}".format(key, matr_tally[key]))
    #"""
    print("     ")
    print("# ********************************")
    print("     ")
    print("Redefining Adinkra classes in sign-permutation space")
    print("     ")
    
    #from collections import OrderedDict
    #class_dict = OrderedDict(sorted(class_dict.items(), key=lambda k: k))
    for key, value in class_dict.items():
        print("\n{} -> \n".format(key))
        for elem in value:
            print("    {}".format(elem))
    #"""
    """
    print("     ")
    print("# ********************************")
    print("     ")
    print("Exporting Adinkra classes as graphics in sign-permutation space")
    print("     ")

    #draw_classes(class_matr) 
    """

    #f.close()

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
# Calculate all matches per a 12 match group.
def build_four_pillars(zeropt_mat, doz_mats):

    idt_mat = 2 * np.matlib.identity(4, dtype=int)
    bakers_doz = doz_mats

    pt0_mat = zeropt_mat[1]
    pt0_mnum = zeropt_mat[0]
    tetnum_list = []

    """ ark_12of4_pairs is a dict that stores
        the index number of 12 matrices that match with the 1st one
        as keys. For each 12 matrix index numbers it then saves
        the index numbers of 4 other matrices out of the 12 that pairs
        match satisfying equation
    """
    ark_12of4_pairs = {}

    for i, li in enumerate(bakers_doz):
        # temp = []
        temp_check = []
        num_li = li[0]
        matli = li[1]
        if (num_li) not in ark_12of4_pairs:
            ark_12of4_pairs['%i' % num_li] = []

        ###print("Finding pairs within the 12 matrices, i = ",num_li)
        for j, lj in enumerate(bakers_doz):
            matlj = lj[1]
            num_lj = lj[0]
            ri = np.transpose(matli)
            rj = np.transpose(matlj)

            if i == j:
                continue
                # if np.array_equal(ri,rj):
                #     tmat = np.matmul(matli,ri) + np.matmul(matli,ri)
                #     if np.array_equal(tmat, idt_mat):
                #         # print("EQ 2.4a satisfied for I = J ",num_li, num_lj)
                #         pass
                #     else:
                #         print("FAILURE EQ 2.4a not satisfied\n", tmat, num_lj)
            elif i != j:
                tmat = np.dot(matli,rj) + np.dot(matlj,ri)
                rtmat = np.dot(ri,matlj) + np.dot(rj,matli)
                if np.count_nonzero(rtmat) == 0:
                    if np.array_equal(ri, inv(matli)) and np.array_equal(rj, inv(matlj)):
                        ark_12of4_pairs[str(num_li)].append(num_lj)
                        ###print("Matching I J pair: ",num_li,num_lj)
                    # if (ri==inv(matli)).all() and (rj==inv(matlj)).all():
                        # temp_check.append((matli,matlj))
                elif np.count_nonzero(rtmat) != 0:
                    pass
                    # print("Not matching I J pair: ",num_li,num_lj)
    """ Build all the possible tetrad combinations """
    for key, four_pairs in ark_12of4_pairs.items():
        """ Hold 3 matrix numbers """
        temp_tri = []
        ###print("Building tetrad combinations")
        ###print(pt0_mnum, key, four_pairs)
        # Temp list for storing local tetrads for each of ix pairs.
        local_tetrad = []
        for oneof4 in four_pairs:
            temp_tri.append((pt0_mnum))
            temp_tri.append(int(key))
            # temp_tri.append((int(key),bakers_doz[int(key)-1][0]))
            i_4temp = ark_12of4_pairs[str(oneof4)]
            # print("Matching pairs for i:", oneof4, i_4temp)
            s = set(four_pairs)
            temp_tri.append(oneof4)

            # Ixpairs is a list of matrices from the 12 matrices
            # that pair with each other and with any 1 of the 12
            # This is necessary to construct a tetrad, the 1st member
            # of the tetrad is always going to be the 0 matrix, and then
            # the other 3 matrices come from the list of 12, provided that
            # all 3 pair with each other as well.
            ixpairs = [m for m in i_4temp if m in s]
            # print("Pair matches for", key, "and ", oneof4, ixpairs)
            # print("Matches for:",oneof4,"of the 4")
            # print(key, oneof4, ixpairs)
            lt = []
            for m in ixpairs:
                # lt = []
                # print(key, oneof4, temp_tri)
                lt.extend(temp_tri)
                lt.append(m)
                lt.sort()
                lt_perm_list = []

                if lt not in local_tetrad:
                    local_tetrad.append(lt)
                    lt_perm_list = list(itertools.permutations([lt[0],lt[1],lt[2],lt[3]],4))
                    # print("Length of permutations for ",m," = ", len(lt_perm_list))
                # if lt not in tetnum_list:
                    # tetnum_list.append(lt)
                for ltx in lt_perm_list:
                    if ltx not in tetnum_list:
                        tetnum_list.append(ltx)
                    else:
                        pass
                lt = []
            """ Wipe the temp_tri for next matrices in the four_pairs """
            temp_tri = []
            # print("Number of unique tetrads:", len(tetnum_list))
    ###print(len(tetnum_list))
    # for mx in tetnum_list:
        # print(mx)
    return tetnum_list

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


# ******************************************************************************
# Generate all sign permutations of an nxn Identity Matrix
def gen_signm(n):
    # itemsl = np.ones(n)
    # items = itemsl.tolist()
    items = [1] * 4

    # ptemp = []
    sign_mat = []
    for signs in itertools.product([-1,1], repeat=len(items)):
        temp = np.array([a*sign for a,sign in zip(items,signs)],dtype=int)
        # ptemp.append(temp)
        sign_mat.append(np.diag(temp))

    # for x in ptemp:
    #     temp = np.diag(x)
    #     sign_mat.append(temp)

    with open('sign_permutation_matrices.txt', 'w') as outfile:
        for smat in sign_mat:
            # outfile.write(smat.astype(str))
            outfile.write(np.array_str(smat))
            outfile.write("\n")
            outfile.write("\n")
    return sign_mat


# ******************************************************************************
# def plusAndMinusPermutations(items):
def plusAndMinusPermutations(items):

    combo_l = []
    # for p in itertools.permutations(items):
    for signs in itertools.product([-1,1], repeat=len(items)):
        # yield [a*sign for a,sign in zip(items,signs)]
        temp = np.array([a*sign for a,sign in zip(items,signs)])
        combo_l.append(temp)
    return combo_l


# ******************************************************************************
# Function for nxn matrices, P(l) - matrix rep. of permutation matrices
def gen_dpermut_mat():
    temp = []

    # perm_rep_list = list(itertools.permutations([0,1,2,3,4,5,6,7,8,9,10,11],4))
    perm_rep_list = list(itertools.permutations([0,1,2,3],4))
    # t1 = np.array((1,0,0,0))
    t1 = np.array((1,0,0,0))
    t2 = np.array((0,1,0,0))
    t3 = np.array((0,0,1,0))
    t4 = np.array((0,0,0,1))
    pml = [t1,t2,t3,t4]
    for mi in perm_rep_list:
        temp_mat = numpy.column_stack((pml[mi[0]],pml[mi[1]],pml[mi[2]],pml[mi[3]]))
        matint = temp_mat.astype(int)
        temp.append(matint)

    newlist = []
    for i in temp:
        if not any(np.array_equal(i, xmat) for xmat in newlist):
            newlist.append(i)

    with open('dxd_permutation_matrices.txt', 'w') as outfile:
        for pmat in temp:
            # outfile.write(smat.astype(str))
            outfile.write(np.array_str(pmat))
            outfile.write("\n")
            outfile.write("\n")
    # for i, x in enumerate(newlist):
    #     print(i, x, "\n")
    # print(len(newlist))
    return(temp)

# ******************************************************************************
# Do the final Vij calculation
def calculate_posvij_matrices(main_tetrad_ark):

    """ Remember that the main_tetrad_ark is a list of lists,
        with each list containing four tuples, with tuples being
        matrix number and the matrices itself. """

    # Import all the possible solutions to the Vij matrices
    vij_possibilities = matrix_outerprod_calc.illuminator_of_elfes()
    vij_matrices = []

    print("                            ")
    print("    Calculating Vij matrices")
    print("                            ")
    # for i in range(0, len(main_tetrad_ark)):
    for i in range(0, len(vij_possibilities)):
        tet_i = [x[1] for x in main_tetrad_ark[i]]
        tri_tet = [np.transpose(i) for i in tet_i]
        print("# ********************************")
        # print("                                     ")
        print("MATRIX i: ", i)
        print("                                     ")
        for j in range(0, len(main_tetrad_ark)):
            tet_j = [x[1] for x in main_tetrad_ark[j]]
            trj_tet = [np.transpose(j) for j in tet_j]
            vij_temp = []
            # print("# ********************************")
            print("        ")
            print("MATRIX j: ", j)
            temp_zero = np.zeros((4,4), dtype=int)
            for x in range(0,len(tet_i)):
                test_1half = np.dot(tri_tet[x],tet_j[x])
                test_2half = np.dot(trj_tet[x],tet_i[x])
                test_difs = np.subtract(test_1half, test_2half)
                # print(" ")
                # print(test_difs)
                temp_mat = np.dot(tri_tet[x],tet_j[x]) - np.dot(trj_tet[x],tet_i[x])
                vij_temp.append(temp_mat)
                # print("")
            temp_add1 = np.add(vij_temp[0], vij_temp[1])
            temp_add2 = np.add(temp_add1, vij_temp[2])
            tempf = np.add(temp_add2, vij_temp[3])
            # tempf = np.divide(temp_add3, 2)
            for ijx in vij_possibilities:
                if np.array_equal(temp_addf, ijx[0]):
                    print("*************$$$$$$$$$$$$$$$$$$***************** ")
                    print("l-solution found:", ijx[1])
                    print(temp_addf)
                    print("")
                    print(ijx[0])
            if  np.array_equal(temp_addf, temp_zero):
                pass
            else:
                vij_matrices.append(temp_addf)
            # print("")
            print(temp_addf)
            # vij_matrices.append(temp_addf)
        vijmats_size = sys.getsizeof(vij_matrices)
        print("Size of Vij Matrices list: bytes / kilobytes:", vijmats_size, vijmats_size/1024)
    print("Length of Vij Matrices")
    print(len(vij_matrices))
    print(vij_matrices)
    pass

# **************************************************************************
# Run main()
if __name__ == "__main__":
    start_time = time.time()

    main()
    print("-- Execution time --")
    print("---- %s seconds ----" % (time.time() - start_time))
