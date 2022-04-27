#!/usr/bin/env python
# coding: utf-8

# In[1001]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy
import time
from tqdm import tqdm_notebook as tqdm

import holoviews as hv
import bebi103
hv.extension('bokeh')


# In[2]:


def product(*args, repeat=1):
    """modified code from Python Software Foundations description of itertools' product function,
    product produces the "Cartesian product of input iterables."""
    
    pools = [pool for pool in args] * repeat
    result2 = [[]]
    for pool in pools:
        result2 = [x+[y] for x in result2 for y in pool]

    for prod in result2:
        yield prod


# In[3]:


def product_index(*args, repeat=1):
    """modified code from Python Software Foundations description of itertools' product function,
    product_index tracks the indices of a given product"""
    
    pools = [pool for pool in args] * repeat
    result2 = [[]]
    for pool in pools:
        result2 = [x+[j] for i, x in enumerate(result2) for j, y in enumerate(pool)]

    for prod in result2:
        yield prod


# In[4]:


def cross(mother, father):
    """cross generates the genotypes for each offspring from the cross between mother and father"""
    
    # o_d = offspring_diploidloci, lists of all combinations of alleles for each loci
    # o_g = offspring_genotypes, all possible offspring genotypes generated from o_d
    # o_d_i = offspring_diploidloci_indices, tracks allele lineage for each locus
    # o_g_i = offspring_genotypes_indices, tracks allele lineage for each genotype
        
    o_d = [list(product(m_a, f_a)) for f_a, m_a in zip(father, mother)]

    o_g = list(product(*o_d))

    o_d_i = [list(product_index(m_a, f_a)) for f_a, m_a in zip(father, mother)]

    o_g_i = list(product(*o_d_i))
    
    
    return(o_g, o_g_i)


# In[5]:


def allele_list(genotype, list_type):
    """allele_list generates a single list of alleles from the input genotype"""
    
    # list_type determines whether output is as a genotype or haplotypes
    # p1 determines which parent's haplotype is the primary one
    
    # g_l = genotype_list, simple list of all alleles for specified genotype
    # mo_h = mother_haplotype, list of alleles from mother
    # fa_h = father_haplotype, list of alleles from father

    if list_type == 'genotype':

        g_l = []

        for locus_index, locus in enumerate(genotype):
            for allele_index, allele in enumerate(locus):
                g_l.append(genotype[locus_index][allele_index])

        return(g_l)

    elif list_type == 'haplotype':

        mo_h = []
        fa_h = []

        for locus_index, locus in enumerate(genotype):
            mo_h.append(genotype[locus_index][0])
            fa_h.append(genotype[locus_index][1])

        return(mo_h, fa_h)


# In[6]:


def all_option(subset, overset, replacement = 0):
    """Determines whether all elements of the subset are in the overset, either with (replacement = 1) or
    without replacement (= 0)"""
    
    subsetcopy = subset.copy()
    oversetcopy = overset.copy()
    
    if replacement:
        check = 1
        for item in subsetcopy:
            if not item in oversetcopy:
                check = 0
                break
    
        return(check)
    
    else:
        check = 1
        for item in subsetcopy:
            if item in oversetcopy:
                oversetcopy.remove(item)
                
            else:
                check = 0
                break
    
        return(check)


# In[7]:


def drive_Sterility(sterile, fertile, P_g_l):
    """To be reimplemented later, adresses sterility as a posisble input. In principle takes into account
    genotype decided 'sterilities' (such as a homozygous mutant causing sterility, as opposed to an allele 
    causing a small reduction to fertility)"""
    
    Psc = [None]*len(P_g_l)
    
    for a_l_i, a_l in enumerate(P_g_l):
        sc = 0
        for index, element in enumerate(sterile):
#             print(lethal)
            if all_option(element, a_l, 0):
                for r_a in fertile[index]:
#                     print(rescue)
                    if all_option(r_a, a_l, 0):
                        sc = 0
                        break
                    else:
                        sc = 1
            
        Psc[a_l_i] = sc
    
    return(np.array(Psc))


# In[8]:


def sterility_cost(alleles_sc, Psc, P_g_l, P_a, SC_type = 'dominant'):
    """To be reimplemented later, adresses sterility as a posisble input. In principle takes into account
    alleles causing a small reduction to fertility"""
    
    Psc_new = [1]*len(Psc)
    
    if SC_type == 'dominant':
        for P_i, P_g in enumerate(P_g_l):
            for a_i, allele in enumerate(P_a):
                added_sc = (allele in P_g)*alleles_sc[a_i]

                if Psc_new[P_i] - Psc[P_i] - added_sc <= 0:
                    Psc_new[P_i] = 0
                else:
                    Psc_new[P_i] -= Psc[P_i] + added_sc
    elif SC_type == 'additive':
        for P_i, P_g in enumerate(P_g_l):
            for a_i, allele in enumerate(P_a):
                added_sc = P_g.count(allele)*alleles_sc[a_i]

                if Psc_new[P_i] - Psc[P_i] - added_sc <= 0:
                    Psc_new[P_i] = 0
                else:
                    Psc_new[P_i] -= Psc[P_i] + added_sc
    return(Psc_new)


# In[9]:


def fitness_cost(f_c, geno_alleles, all_alleles):
    """fitness_cost takes into account two distinct sources of fitness affects and combines them:
    - fitness affects from single copies of alleles that can be dominant or additive
    - fitness affects from 'genotype' affects (wherein a heterozygote might be neutral but a homozygote
    for a given allele is a lethal condition)"""
    
    lethal = f_c[2]
    rescue = f_c[3]
    
    if isinstance(f_c[0][0], str):
        f_c_total = [np.zeros(len(geno_alleles[0])), np.zeros(len(geno_alleles[1]))]
    
        for sex in [0,1]:

            for g_i, genotype in enumerate(geno_alleles[sex]):

                for index, element in enumerate(lethal):
                    if all_option(element, genotype, 0):
                        for r_a in rescue[index]:
                            if all_option(r_a, genotype, 0):
                                break
                            else:
                                f_c_total[sex][g_i] = 1

                if f_c_total[sex][g_i] == 0:
                    added_fc = 0
                    for a_i, allele in enumerate(all_alleles):
                        if f_c[0][a_i] == 'dominant':
                            added_fc += (allele in genotype)*f_c[1][a_i]
                        elif f_c[0][a_i] == 'additive':
                            added_fc += genotype.count(allele)*f_c[1][a_i]
                        elif f_c[0][a_i] == 'multiplicative':
                            added_fc += 'later'
                        else:
                            return('Throw an error here')

                    if f_c_total[sex][g_i] + added_fc >= 1:
                        f_c_total[sex][g_i] = 1
                    else:
                        f_c_total[sex][g_i] += added_fc
                        
    if isinstance(f_c[0][0], float):
        f_c_total = [f_c[0], f_c[1]]
        
        for sex in [0,1]:
            for genotype in genotypes:
                geno_alleles[sex].append([a for l in genotype for a in l])

            for g_i, genotype in enumerate(geno_alleles[sex]):

                for index, element in enumerate(lethal):
                    if all_option(element, genotype, 0):
                        for r_a in rescue[index]:
                            if all_option(r_a, genotype, 0):
                                break
                            else:
                                f_c_total[sex][g_i] = 1
            
    return(f_c_total)


# In[ ]:


"Test case for a four locus ClvR scenario, definitely don't run this one yet"
# alleles = [['T', 'A'], ['W', 'C', 'R'], ['D', 'M'], ['X', 'Y', 'Yt']]
# mods = [['germline', ['T'], 'W', ['C', 'R']],
#         ['zygotic', [['Yt'], [''], ['']], 'W', ['C']],
#         ['zygotic', [['T'], [''], ['']], 'W', ['C', 'R']],
#         ['somatic', ['T', 'T'], 'Y', ['Yt']]
#        ]
# sex_det = ['XY']


# In[ ]:


"Test case for a three locus ClvR scenario,  don't run this one yet"
# alleles = [['T', 'A'], ['W', 'C', 'R'], ['X', 'Y', 'Yt']]
# mods = [['germline', ['T'], 'W', ['C', 'R']],
#         ['zygotic', [['T'], [''], ['']], 'W', ['C', 'R']],
#         ['somatic', ['T', 'T'], 'Y', ['Yt']]
#        ]
# sex_det = ['XY']


# In[ ]:


"Simpler test case for a three locus ClvR scenario, don't run this one yet"
# alleles = [['T', 'A'], ['W', 'C', 'R'], ['X', 'Y']]
# mods = [['germline', ['T'], 'W', ['C']],
#         ['zygotic', [['T'], [''], ['']], 'W', ['C']]
#        ]
# sex_det = ['XY']


# In[10]:


"Test case for a two locus ClvR scenario, when everything is working this is a first simple test"
alleles = [['T', 'A'], ['Xc', 'X', 'Y']]
mods = [['germline', ['T'], 'X', ['Xc']],
        ['zygotic', [['T'], [''], ['']], 'X', ['Xc']]
       ]
sex_det = ['XY', [], []]


# In[323]:


# mother = [['T', 'T'], ['X', 'X']]
# father = [['T', 'T'], ['X', 'Y']]

mother = [['T', 'T']]
father = [['T', 'T']]

offspring, offspring_inds = cross(mother, father)
print(offspring_inds)


# In[874]:


def cross_matrix_generator(alleles, mods, sex_det):
    """cross_matrix_generator produces the cross matrices and other associated information for doing a 
    population dynamics simulation from a user defined set of instructions to define a gene drive.
    Note the explicit input structure:
    - alleles is a nested set of lists, each inner list is the alleles available at that locus
    - mods is a nested set of lists, each inner list are the instructions for a single gene drive behavior containing:
        - drive timing as a str, options are 'germline', 'zygotic', and 'somatic'
        - alleles needed to cause this drive behavior, either as a list or list of lists
            - the list of lists is needed for 'zygotic' behvaior, lists represent alleles needed from the mother, 
            father, and offspring, respectively, to cause said drive behavior
        - drive target allle as a str
        - list of posssible alleles that the target allele can be changed to as a result of drive behavior
    - sex_det is a list which specifies:
        - the type of sex/sex alleles as a str
            - 'XY', 'ZW', 'autosomal', 'plant XY', 'plant autosomal'
        - a list of lists of alleles that can cause an individual to be female beyond normal XX or ZW rules
        - a list of lists of alleles that can cause an individual to be male beyond normal XY or ZZ rules
    """
    
    # First, generate  basic info like number of loci and all possible genotypes, then filter out the nonsense.
    # genotypes consists of three sets of "genotype lists" and three sets of "allele lists", each with 
    # all individuals, females, and males, respectively. Both are provided because genotype lists are the best for
    # performing crosses, but its much simpler to search through the allele lists to check fro certain alleles
    
    num_loci = len(alleles)
    all_alleles = [allele for locus in alleles for allele in locus]
    
    diploid_loci = [list(product(allele, allele)) for allele in alleles]
    genotypes_raw = list(product(*diploid_loci))
    
    nonsense_genotypes = []
    genotypes = [[], [], [], [], [], []]
    
    if sex_det[0] == 'autosomal':
        genotypes[0] = genotypes_raw
        genotypes[1] = genotypes_raw
        genotypes[2] = genotypes_raw
        for genotype in genotypes_raw:
            genotypes[3].append([a for l in genotype for a in l])
            genotypes[4].append([a for l in genotype for a in l])
            genotypes[5].append([a for l in genotype for a in l])
        
    elif sex_det[0] == 'XY':
        for genotype in genotypes_raw:
            if "Y" in genotype[-1][0]:
                nonsense_genotypes.append(genotype)
                
            else:
                genotypes[0].append(genotype)
                genotypes[3].append([a for l in genotype for a in l])
                
                if any([all_option(sexing, genotypes[3][-1]) for sexing in sex_det[1]]):
                    genotypes[1].append(genotype)
                    genotypes[4].append([a for l in genotype for a in l])
                    
                elif any([all_option(m_c, genotypes[3][-1]) for m_c in sex_det[2]]):
                    genotypes[2].append(genotype)
                    genotypes[5].append([a for l in genotype for a in l])
                        
                elif 'Y' in genotype[-1]:
                    genotypes[2].append(genotype)
                    genotypes[5].append([a for l in genotype for a in l])
                    
                else:
                    genotypes[1].append(genotype)
                    genotypes[4].append([a for l in genotype for a in l])
        
    elif sex_det[0] == 'ZW':
        # later
        later = []
        
    elif sex_det[0] == 'plant XY':
        # later
        later = []
        
    elif sex_det[0] == 'plant autosomal':
        # later
        later = []
        
    else:
        return('Throw error message here')
    
    # info for dealing with recombination distances, namely the total number of them and the length of their 
    # corresponding dimension in the cross matrix (always the "fourth dimension", ie the third index)
    n_r_d = num_loci-1
    l_r_d_d = 2**(n_r_d*2)
    
    # dictionary for handling cross information and how that translates to an index in the cross matrices.
    # definitely bugged, I am actively working on this at the moment so Jackie and Roy don't stress this section
    mod_dict = {}
    mod_dim_max_ind = 3

    for mod in mods:
        if (mod[0], mod[2]) in mod_dict:
            mod_dim_counter = mod_dict[(mod[0], mod[2])][0]

#         else:
#             if mod[0] == 'germline':
#                 mod_dim_max_ind += 2
#                 mod_dict[mod_dim_max_ind-1] = (mod[0], mod[2])
#             else:
#                 mod_dim_max_ind += 1
#             mod_dim_counter = 1
#             mod_dict[mod_dim_max_ind] = (mod[0], mod[2])
        else:
            mod_dim_counter = 1
            mod_dim_max_ind += 1
            mod_dict[mod_dim_max_ind] = (mod[0], mod[2])
            mod_dim_max_ind += 1
            mod_dict[mod_dim_max_ind] = (mod[0], mod[2])
            

        for sub_mod in mod[3]:
            mod_dict[(mod[0], mod[2], sub_mod)] = mod_dim_counter
            mod_dim_counter += 1

        # Add updated dimension counter with addition of "drive failure" sub_mod
        mod_dict[(mod[0], mod[2], 'drive failed')] = mod_dim_counter
        mod_dict[(mod[0], mod[2])] = [mod_dim_counter + 1, mod_dim_max_ind]
    print(mod_dict)
    
    # Define the dimension lengths of the cross_matrices, female cm first followed by male cm
    # First four dimensions: 'mother genotypes', 'father genotypes', 'offspring genotypes', 'recombination identity'
    c_m_temp = [[len(genotypes[1]), len(genotypes[2]), len(genotypes[1]), l_r_d_d],
           [len(genotypes[1]), len(genotypes[2]), len(genotypes[2]), l_r_d_d]]
    
    # Add additional dimensions for each unique drive-type and target allele combo
    for mod_dim in range(4, mod_dim_max_ind+1):
        c_m_temp[0].append(mod_dict[mod_dict[mod_dim]][0])
        c_m_temp[1].append(mod_dict[mod_dict[mod_dim]][0])
    
    # Make actual cross_matrices as np arrays using specified dimension lengths
    c_m = [np.zeros(c_m_temp[0]), np.zeros(c_m_temp[1])]
    print(c_m[0].shape)
    
    # Fill in cross_matrices with the actual cross information based on drive based modificiation
    # Jackie and Roy follow what you can but I haven't detailed all the variables yet because I am still 
    # working on this section so don't worry too much. For some reference, 'r_d' refers to recombination 
    # distance to refer to the effects of recombination events, 'sprin' is a shorthand for offspring, 'ind' for index,
    # 'dim' or 'd' at the end of a variable name usually refers to a dimension index for the cross matrices
    for mo_ind, mother in enumerate(genotypes[1]):
        for fa_ind, father in enumerate(genotypes[2]):
#             print('Cross = ' + str(mother) + 'x' + str(father))
            mother_alleles = genotypes[4][mo_ind]
            father_alleles = genotypes[5][fa_ind]
            offspring, offspring_inds = cross(mother, father)
            
            for sprin_ind, sprin in enumerate(offspring):
                mod_sprin_list = [sprin]
                sprin_inds = offspring_inds[sprin_ind]
                r_d_ind = ''
                
                if n_r_d > 0:
                    for r_event in range(n_r_d):
                        if sprin_inds[r_event][0] == sprin_inds[r_event+1][0]:
                            r_d_ind += '0'
                        else:
                            r_d_ind += '1'

                    for r_event in range(n_r_d):
                        if sprin_inds[r_event][1] == sprin_inds[r_event+1][1]:
                            r_d_ind += '0'
                        else:
                            r_d_ind += '1'
                
                else:
                    r_d_ind = '0'
                
                base_sprin_mod_ind = [genotypes[1].index(mother), genotypes[2].index(father), 0, int(r_d_ind, 2)]
                
                for mod_dim in range(4, mod_dim_max_ind+1):
                    base_sprin_mod_ind.append(0)
                    
                sprin_mod_ind = [base_sprin_mod_ind]
                
                for m_s_ind, mod_sprin in enumerate(mod_sprin_list):
                    
                    mod_sprin_a_l = allele_list(mod_sprin, 'genotype')
#                     print('sprin = ' + str(mod_sprin) + '-' + str(sprin_mod_ind[m_s_ind]))
                    if mods != []:
                        for mod in mods:
                            if mod[0] == 'germline':
                                if mod[2] in mod_sprin_a_l:
                                    mod_a_ind = mod_sprin_a_l.index(mod[2])
                                    if all_option(mod[1], mother_alleles) and mod_a_ind%2 == 0:
                                        sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], 
                                                                                                           'drive failed')]
                                        for sub_mod in mod[3]:
                                            temp_sprin = copy.deepcopy(mod_sprin)
                                            temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                            mod_sprin_list.append(temp_sprin)

                                            temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                            temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0],mod[2], sub_mod)]
                                            sprin_mod_ind.append(temp_inds)
    #                                         print(sub_mod+'-'+str(temp_inds)+'-mother')

                                    elif all_option(mod[1], father_alleles):
                                        if mod_a_ind%2 == 1 or mod_sprin_a_l[mod_a_ind+1] == mod[2]:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], 
                                                                                                             'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0],mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)
    #                                             print(sub_mod+'-'+str(temp_inds)+'-father')

                            # Likely need to exclude these from subsequent zygotic modification
                            elif mod[0] == 'zygotic':
                                if mod[2] in mod_sprin_a_l:
                                    mod_a_ind = mod_sprin_a_l.index(mod[2])
                                    if (all_option(mod[1][0], mother_alleles) and 
                                        all_option(mod[1][1], father_alleles) and
                                        all_option(mod[1][2], mod_sprin_a_l)):
                                        if mod_a_ind%2 == 0 and mod_sprin_a_l[mod_a_ind+1] == mod[2]:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], 
                                                                                                               'drive failed')]
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], 
                                                                                                             'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                        elif mod_a_ind%2 == 0:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], 
                                                                                                               'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0],mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                        elif mod_a_ind%2 == 1:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], 
                                                                                                             'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                            # Likely need to exclude these from subsequent somatic modification
                            elif mod[0] == 'somatic':
                                if mod[2] in mod_sprin_a_l:
                                    mod_a_ind = mod_sprin_a_l.index(mod[2])
                                    if all_option(mod[1], mod_sprin_a_l):
                                        if mod_a_ind%2 == 0 and mod_sprin_a_l[mod_a_ind+1] == mod[2]:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], 
                                                                                                               'drive failed')]
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], 
                                                                                                             'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                        elif mod_a_ind%2 == 0:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0],mod[2], 
                                                                                                             'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][0] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]-1] = mod_dict[(mod[0], mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)

                                        elif mod_a_ind%2 == 1:
                                            sprin_mod_ind[m_s_ind][mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0],mod[2], 
                                                                                                             'drive failed')]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[int(mod_a_ind/2)][1] = sub_mod
                                                mod_sprin_list.append(temp_sprin)

                                                temp_inds = copy.deepcopy(sprin_mod_ind[m_s_ind])
                                                temp_inds[mod_dict[(mod[0],mod[2])][1]] = mod_dict[(mod[0],mod[2], sub_mod)]
                                                sprin_mod_ind.append(temp_inds)
                
                # This will be adjusted with varying sex_mods, for now sorts female and male offspring
                for mod_sprin_ind, mod_sprin in enumerate(mod_sprin_list):
                    if sex_det[0] == 'XY':
                        if mod_sprin in genotypes[1]:
                            sprin_mod_ind[mod_sprin_ind][2] = genotypes[1].index(mod_sprin)
                            c_m[0][tuple(sprin_mod_ind[mod_sprin_ind])] += 1

                        elif mod_sprin in genotypes[2]:
                            sprin_mod_ind[mod_sprin_ind][2] = genotypes[2].index(mod_sprin)
                            c_m[1][tuple(sprin_mod_ind[mod_sprin_ind])] += 1

                        else:
                            print('We have a problem offspring')
                            
                    elif sex_det[0] == 'autosomal':
                        sprin_mod_ind[mod_sprin_ind][2] = genotypes[0].index(mod_sprin)
                        c_m[0][tuple(sprin_mod_ind[mod_sprin_ind])] += 1
                        c_m[1][tuple(sprin_mod_ind[mod_sprin_ind])] += 1
#                         print('Sprin-added: ' + str(mod_sprin) + str(sprin_mod_ind[mod_sprin_ind]))
                    
    return(c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict)


# In[878]:


def drive_simulation(c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict, 
                     num_gens, d_a, r_d, f_c, intro = [[0.3], [0]], 
                     sim_type = 'deterministic', c_m_type = 'matrix', 
                     m_r = 0.01, a_r = [0, [0.3], [0], 200, 50]):
    
    """
    performs a population dynamics simulation based on the input cross matrices and specific variables for:
    - num_gens = length of simulation in generations
    - d_a = drive activity, in order of drive behaviors lists in mods and icnluding rates for each mod 
    ('unmodified' rate is calculated by subtracting the sum of the options from 1 for each behavior)
    - r_d = recombination distance info, still being updated
    - f_c = fitness cost information, overlist containing:
        - list of strings describing fitness type for each allle
            - 'dominant', 'additive', or 'multiplicative'
        - list of floats for ftiness costs per allele in order of all_alleles list
        - list of lists of strs, designates allelels that compose lethal conditions
        - list of lists of strs, designates allelels that compose rescue conditions per lethal conditon
    - sim_type sets whether it is a deterministic or stochastic simulation
    - c_m_type is archaic, would have distinguished between matrix vs old string method of making cross matrices
    
    intro: information about initial release of gene drive bearing transgenics
            - list of release frequencies/numbers for females and males
                - single item list for an even split between females and males for a single genotype per sex,
                single two item list for specifying female and male frequencies for a single genotype per sex, 
                two lists for frequencies of multiple released genotypes per sex
            - list of release genotypes for females and males
                - single item list for release of the same corresponding genotype from both sexes (index-wise),
                single two item list for release of a distinct female genotype and a distinct male genotype, 
                two lists for release of multiple distinct female genotypes and male genotypes
    
    a_r: additional release information. Indices in order are: 
            - number of additional releases
            - list of additional release frequencies/numbers for females and males
                - same rules as per intro release frequencies/numbers
            - list of additional release genotypes for females and males
                - same rules as per intro release genotypes
            - starting generation for additional releases
            - number of generations between additional releases
    
    """
    
    n_f_g = len(genotypes[1])
    n_m_g = len(genotypes[2])
    
    geno_alleles = genotypes[4:]
    
    FC = fitness_cost(f_c, geno_alleles, all_alleles)
    
    if c_m_type == 'matrix':
        
        # Set recombination rates based on recombination distances for females and males
        if isinstance(r_d[0], int) and len(r_d) == n_r_d:
            r_d_temp = [float(r) for r in r_d]
            r_d = [r_d_temp, r_d_temp]
        elif isinstance(r_d[0], float) and len(r_d) == n_r_d:
            r_d_temp = r_d
            r_d = [r_d_temp, r_d_temp]
        
        elif isinstance(r_d[0], list):
            if len(r_d[0]) == len(r_d[1]) == n_r_d:
                err = 'NA'
            else:
                err = 'throw error: must have the same number of recombination rates for each sex'
        
        else:
            err = 'throw error: r_d must be either a single list or a pair of lists of             length equal to the number of recombination distances'
        
        
        r_d_full = []
        if n_r_d > 0:
            for r_ind, rec in enumerate(r_d):
                r_d_temp = []
#                 [r_d_temp.append([0.5 + 0.5*r, 0.5*(1-r)]) for r in rec]
                [r_d_temp.append([0.5 + 0.5*(1-r/50), 0.5*(r/50)]) for r in rec]
                r_d_full.append(r_d_temp)

            for rec in r_d_full[0:2]:
                r_d_full.append(list(product(*rec)))

            for rec in r_d_full[2:4]:
                r_d_full.append([np.prod(rd) for rd in rec])

            r_d_full.append(list(product(*r_d_full[4:6])))
            r_d_full.append([np.prod(rd) for rd in r_d_full[6]])
        else:
            r_d_full = [[1]]*8
        
        # dimensional factors will hold the rates for each r_d and modification dimension
        # d_b is a drive behavior within all drive activities (d_a)... working on the naming schema
        dimensional_factors = [r_d_full[7]]
        
        if d_a != []:
            for d_ind, d_b in enumerate(d_a):
                if sum(d_b) > 1:
                    error = 'fill in later'

                d_b_full = [1]
                for b in d_b:
                    d_b_full.append(b)
                d_b_full.append(1-sum(d_b))

                if mod_dict[d_ind+4][0] == 'germline':
                    dimensional_factors.append(d_b_full)
                elif mod_dict[d_ind+4][0] == 'zygotic' or mod_dict[d_ind+4][0] == 'somatic':
                    dimensional_factors.append(d_b_full)
                    dimensional_factors.append(d_b_full)
#             print(d_b_full)
        print(dimensional_factors)
        # make the 'whole matrix string' which is a string that builds the executable command to collapse a dimension
        # after it has been multiplied through by all its factors. Ignores the first three indices because those are 
        # specific to mother, father, and offspring genotype, respectively, and that info is processed later
        w_m_s = ["C_M[", "sex", "][:, :, :"]
        C_M = copy.deepcopy(c_m)
        for factor in dimensional_factors:
            w_m_s.append(", :")
        w_m_s.append("] *= ")
        
        cm = [[], []]
        
        for sex in range(2):
            for f_d, factor in enumerate(dimensional_factors):
                for rate_d, rate in enumerate(factor):
                    s_m_s = w_m_s.copy()
                    s_m_s[1] = str(sex)
                    s_m_s[3+f_d] = ", " + str(rate_d)
                    s_m_s += str(rate)
                    s_m = "".join(s_m_s)
                    exec(s_m)
            cm[sex] = np.sum(C_M[sex], axis = tuple(range(3,3+len(dimensional_factors))))
#         print(cm[0].shape)
#         return(cm)
    elif c_m_type == 'string':
        r_d = 'fill in later'
        
    else:
        err = 'throw error: c_m_type must be \'matrix\' or \'string\''
    
    
    # If there are additional releases, generate list of generations at which said releases occur
    if a_r[0] == 0:
        a_r_gens = []
        
    else:
        if isinstance(a_r[0], int) != 1:
            err = 'throw error: must use an integer number of additional releases'
        
        if isinstance(a_r[4], list):
            a_r_gens = a_r[4]
            
        elif isinstance(intro[0][0], int):
            a_r_gens = [a_r[3]]
            for add_rel in range(1, a_r[0]):
                a_r_gens.append(a_r_gens[-1] + a_r[4])
            
        else:
            err = 'throw error: a_r must be either a list of ints or a single int'
    
    
    # Based on simulation type, generate initial generation of the population
    if sim_type == 'deterministic':
        
        females = np.zeros([num_gens+1, len(genotypes[1])])
        males = np.zeros([num_gens+1, len(genotypes[2])])
        
        if isinstance(intro[0][0], list) and isinstance(intro[0][1], list):
            if (isinstance(intro[1][0], list) and 
                isinstance(intro[1][1], list) and 
                len(intro[0][0]) == len(intro[1][0]) and 
                len(intro[0][1]) == len(intro[1][1])):
                
                first_females = np.zeros(len(genotypes[1]))
                for ind, f_geno in enumerate(intro[1][0]):
                    first_females[f_geno] = intro[0][0][ind]
                    
                first_males = np.zeros(len(genotypes[2]))
                for ind, m_geno in enumerate(intro[1][1]):
                    first_males[m_geno] = intro[0][1][ind]
                
                i_f_tot = np.sum([intro[0][0], intro[0][1]])
                
                first_females[-1] = (1-i_f_tot)/2
                first_males[-1] = (1-i_f_tot)/2
                
                females[0] = intro[0][0]
                males[0] = intro[0][1]
            
            else:
                err = 'throw error: length of intro frequencies and genotypes must match for respective sexes'
            
        elif len(intro[0]) == 1:
            if len(intro[1]) == 1:
                females[0, intro[1][0]] = intro[0][0]/2
                males[0, intro[1][0]] = intro[0][0]/2
            
            elif len(intro[1]) == 2:
                females[0, intro[1][0]] = intro[0][0]/2
                males[0, intro[1][1]] = intro[0][0]/2
                
            else:
                 err = 'throw error: can only have one or two specified intro genotypes for one intro frequency'
            
            females[0, -1] = (1-intro[0][0])/2
            males[0, -1] = (1-intro[0][0])/2
            
        elif len(intro[0]) == 2:
            if len(intro[1]) == 1:
                females[0, intro[1][0]] = intro[0][0]
                males[0, intro[1][0]] = intro[0][1]
            
            elif len(intro[1]) == 2:
                females[0, intro[1][0]] = intro[0][0]
                males[0, intro[1][1]] = intro[0][1]
                
            else:
                err = 'throw error: can only have one or two specified intro genotypes for one or two intro frequencies'
                
            females[0, -1] = (1-intro[0][0]-intro[0][1])/2
            males[0, -1] = (1-intro[0][0]-intro[0][1])/2
            
        else:
            err = 'throw error: intro frequencies must be a list of two lists, or a list with one or two items'
        
        sigma = [0]*(num_gens+1)
        
        # Simulate population dynamics
        for gen in range(1,num_gens+1):
            presentcross = np.outer(females[gen-1], males[gen-1])
            
#             print(presentcross)
            
            f_temp = np.zeros(n_f_g)
            m_temp = np.zeros(n_m_g)

            # Generate gross proportions for each genotype
            for f_geno in range(n_f_g):
                f_temp[f_geno] = np.sum(np.multiply(cm[0][:, :, f_geno], presentcross))
#                 print(np.multiply(cm[0][:, :, f_geno],presentcross))
#             return(cm)
            for m_geno in range(n_m_g):
                m_temp[m_geno] = np.sum(np.multiply(cm[1][:, :, m_geno], presentcross))
            
            sigma[gen] = (np.sum(np.multiply(f_temp, np.subtract([1]*len(FC[0]), FC[0]))) 
                          + np.sum(np.multiply(m_temp, np.subtract([1]*len(FC[1]), FC[1]))))

            # Set final, normalized proportions for each genotype, including
            # additiional release if necessary
            if gen in a_r_gens:
                females[gen][:] = np.multiply(f_temp, np.subtract([1]*len(FC[0]), FC[0]))/sigma[gen]
                males[gen][:] = np.multiply(m_temp, np.subtract([1]*len(FC[1]), FC[1]))/sigma[gen]
                tbc = 'later'

            else:
                females[gen][:] = np.multiply(f_temp, np.subtract([1]*len(FC[0]), FC[0]))/sigma[gen]
                males[gen][:] = np.multiply(m_temp, np.subtract([1]*len(FC[1]), FC[1]))/sigma[gen]
#                 print(gen)
#                 print(np.sum(np.multiply(f_temp, np.subtract([1]*len(FC[0]), FC[0]))))
#                 print(np.sum(np.multiply(m_temp, np.subtract([1]*len(FC[1]), FC[1]))))
#                 print(np.sum(np.multiply(f_temp, np.subtract([1]*len(FC[0]), FC[0])))
#                      + np.sum(np.multiply(m_temp, np.subtract([1]*len(FC[1]), FC[1]))))
#                 print(np.sum(females[gen][:]))
#                 print(np.sum(males[gen][:]))
#                 print('Sum of females and males this gen is: '+ str(np.sum(females[gen][:]) + np.sum(males[gen][:])))
#                 print(sigma[gen])
                
#         sim_females = pd.DataFrame(females, columns=genotypes[4])
#         sim_males = pd.DataFrame(males, columns=genotypes[5])
        return([females, males, cm])
        
    else:
        # this is for the stochastic simulation, not finished until issues in deterministic are sorted
        females[gen] = []
        males[gen] = []
        
        for mo_g_i in females[gen-1]:
            
            new_females = []
            new_males = []
            
            mother = F_g[mo_g_i]
            fa_g_i = random.choice(males[gen-1])
            father = M_g[fa_g_i]
            
            if M_cm.ndim == 3:
                
                F_cm_weights[fa_g_i][mo_g_i] = [eval(daughter) for daughter in d_p]
                M_cm_weights[fa_g_i][mo_g_i] = [eval(son) for son in s_p]
                
            else:
                later = 'when I\'ll finish this'
                
            offspring = np.random.poisson(fertility*(1-Ffc[mo_g_i])*(1-Mfc[fa_g_i]))
            daughters = np.random.binomial(offspring, 0.5)
            sons = offspring-daughters
            
            new_females.extend([daughter for daughter in random.choices(F_g, weights = F_cm_weights[fa_g_i][mo_g_i], 
                                                                        k=daughters)])
            new_males.extend([son for son in random.choices(M_g, weights = M_cm_weights[fa_g_i][mo_g_i], k=sons)])
            
            female_survivors = [np.random.binomial(1, (1/fertility)
                                                   *(Eq/(len(males[gen-1])
                                                         +len(females[gen-1])))*(1-Ffc[x])) for x in new_females]
            male_survivors = [np.random.binomial(1, (1/fertility)
                                                 *(Eq/(len(males[gen-1])
                                                       +len(females[gen-1])))*(1-Mfc[x])) for x in new_males]
            
            females[gen].extend([new_females[x] for x in range(len(new_females)) if female_survivors[x] != 0])
            males[gen].extend([new_males[x] for x in range(len(new_males)) if male_survivors[x] != 0])
            
        for genotype in range(n_F_g):
            F[gen, genotype] = females[gen].count(genotype)
            
        for genotype in range(n_M_g):
            M[gen, genotype] = males[gen].count(genotype)
    


# In[1059]:


def stochastic_sim(alleles, mods, sex_det, 
                   num_gens, intro, d_a, r_d, f_c, 
                   K = 10000, n_o = 100, g_f = 10, 
                   cross_dict = {}):
    
    num_loci = len(alleles)
    all_alleles = [allele for locus in alleles for allele in locus]
    
    diploid_loci = [list(product(allele, allele)) for allele in alleles]
    genotypes_raw = list(product(*diploid_loci))
    
    nonsense_genotypes = []
    genotypes = [[], [], [], [], [], []]
    
    if sex_det[0] == 'autosomal':
        genotypes[0] = genotypes_raw
        genotypes[1] = genotypes_raw
        genotypes[2] = genotypes_raw
        for genotype in genotypes_raw:
            genotypes[3].append([a for l in genotype for a in l])
            genotypes[4].append([a for l in genotype for a in l])
            genotypes[5].append([a for l in genotype for a in l])
        
    elif sex_det[0] == 'XY':
        for genotype in genotypes_raw:
            if "Y" in genotype[-1][0]:
                nonsense_genotypes.append(genotype)
                
            else:
                genotypes[0].append(genotype)
                genotypes[3].append([a for l in genotype for a in l])
                
                if any([all_option(sexing, genotypes[3][-1]) for sexing in sex_det[1]]):
                    genotypes[1].append(genotype)
                    genotypes[4].append([a for l in genotype for a in l])
                    
                elif any([all_option(m_c, genotypes[3][-1]) for m_c in sex_det[2]]):
                    genotypes[2].append(genotype)
                    genotypes[5].append([a for l in genotype for a in l])
                        
                elif 'Y' in genotype[-1]:
                    genotypes[2].append(genotype)
                    genotypes[5].append([a for l in genotype for a in l])
                    
                else:
                    genotypes[1].append(genotype)
                    genotypes[4].append([a for l in genotype for a in l])
        
    elif sex_det[0] == 'ZW':
        # later
        later = []
        
    elif sex_det[0] == 'plant XY':
        # later
        later = []
        
    elif sex_det[0] == 'plant autosomal':
        # later
        later = []
        
    else:
        return('Throw error message here')
    
    geno_alleles = genotypes[4:]
    FC = fitness_cost(f_c, geno_alleles, all_alleles)
    n_r_d = num_loci-1
    
    adults = [[[]], [[]]]
    
    for individual in range(int(intro[0][0]*K/2)):
        adults[0][0].append(genotypes[0][intro[1][0]])
        adults[1][0].append(genotypes[1][intro[1][0]])
        
    for individual in range(int((1-intro[0][0])*K/2)):
        adults[0][0].append(genotypes[0][-1])
        adults[1][0].append(genotypes[1][-1])
    
    for a in d_a:
        a.append(1 - np.sum(a))
    
    # Set recombination rates based on recombination distances for females and males
    if isinstance(r_d[0], int) and len(r_d) == n_r_d:
        r_d_temp = [float(r) for r in r_d]
        r_d = [r_d_temp, r_d_temp]
    elif isinstance(r_d[0], float) and len(r_d) == n_r_d:
        r_d_temp = r_d
        r_d = [r_d_temp, r_d_temp]

    elif isinstance(r_d[0], list):
        if len(r_d[0]) == len(r_d[1]) == n_r_d:
            err = 'NA'
        else:
            err = 'throw error: must have the same number of recombination rates for each sex'

    else:
        err = 'throw error: r_d must be either a single list or a pair of lists of         length equal to the number of recombination distances'
    
    r_d_full = []
    if n_r_d > 0:
        for r_ind, rec in enumerate(r_d):
            r_d_temp = []
#                 [r_d_temp.append([0.5 + 0.5*r, 0.5*(1-r)]) for r in rec]
            [r_d_temp.append([0.5 + 0.5*(1-r/50), 0.5*(r/50)]) for r in rec]
            r_d_full.append(r_d_temp)

        for rec in r_d_full[0:2]:
            r_d_full.append(list(product(*rec)))

        for rec in r_d_full[2:4]:
            r_d_full.append([np.prod(rd) for rd in rec])

        r_d_full.append(list(product(*r_d_full[4:6])))
        r_d_full.append([np.prod(rd) for rd in r_d_full[6]])
    else:
        r_d_full = [[1]]*8
    
    r_d_ind_list = r_d_full[7]
    
    mod_dict = {}
    mod_dim_max_ind = 0

    for mod in mods:
        if (mod[0], mod[2]) in mod_dict:
            mod_dim_counter = mod_dict[(mod[0], mod[2])][0]
        else:
            mod_dim_counter = 1
            mod_dim_max_ind += 1
            mod_dict[mod_dim_max_ind] = (mod[0], mod[2])
            mod_dim_max_ind += 1
            mod_dict[mod_dim_max_ind] = (mod[0], mod[2])
            

        for sub_mod in mod[3]:
            mod_dict[(mod[0], mod[2], sub_mod)] = mod_dim_counter
            mod_dim_counter += 1

        # Add updated dimension counter with addition of "drive failure" sub_mod
        mod_dict[(mod[0], mod[2], 'drive failed')] = mod_dim_counter
        mod_dict[(mod[0], mod[2])] = [mod_dim_counter + 1, mod_dim_max_ind]
    print(mod_dict)
    
    for gen in tqdm(range(num_gens)):
        adults[0].append([])
        adults[1].append([])
        
        for mother in adults[0][gen]:
            father = random.choice(adults[1][gen])
            
            mother_alleles = genotypes[4][genotypes[1].index(mother)]
            father_alleles = genotypes[5][genotypes[2].index(father)]
            
            if (genotypes[1].index(mother), genotypes[2].index(father)) in cross_dict.keys():
                offspring_modified = cross_dict[(genotypes[1].index(mother), genotypes[2].index(father))]
            
            else:
                offspring, offspring_inds = cross(mother, father)
                offspring_modified = [[], []]

                for sprin_ind, sprin in enumerate(offspring):
                    sprin_inds = offspring_inds[sprin_ind]
                    r_d_ind = ''

                    if n_r_d > 0:
                        for r_event in range(n_r_d):
                            if sprin_inds[r_event][0] == sprin_inds[r_event+1][0]:
                                r_d_ind += '0'
                            else:
                                r_d_ind += '1'

                        for r_event in range(n_r_d):
                            if sprin_inds[r_event][1] == sprin_inds[r_event+1][1]:
                                r_d_ind += '0'
                            else:
                                r_d_ind += '1'

                    else:
                        r_d_ind = '0'

                    sprin_prop = [r_d_ind_list[int(r_d_ind, 2)]]
                    mod_sprin_list = [[sprin, [r_d_ind_list[int(r_d_ind, 2)]]]]
                    
                    for mod_dim in range(1, mod_dim_max_ind+1):
                        mod_sprin_list[0][1].append(1)

                    for mod_sprin in mod_sprin_list:
#                         print(mod_sprin[0])
                        mod_sprin_a_l = allele_list(mod_sprin[0], 'genotype')
    #                     print('sprin = ' + str(mod_sprin) + '-' + str(sprin_mod_ind[m_s_ind]))
                        if mods != []:
                            for mod_i, mod in enumerate(mods):
                                if mod[0] == 'germline' and mod[2] in mod_sprin_a_l:
                                    mod_a_ind = mod_sprin_a_l.index(mod[2])
                                    if all_option(mod[1], mother_alleles) and mod_a_ind%2 == 0:
                                        temp_sprin[1].append(d_a[mod_i][-1])
                                        mod_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][-1]
                                        for sub_mod in mod[3]:
                                            temp_sprin = copy.deepcopy(mod_sprin)
                                            temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                            temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                            mod_sprin_list.append(temp_sprin)
#                                             print(sub_mod+'-'+str(temp_inds)+'-mother')

                                    elif all_option(mod[1], father_alleles):
                                        if mod_a_ind%2 == 1 or mod_sprin_a_l[mod_a_ind+1] == mod[2]:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)
    #                                             print(sub_mod+'-'+str(temp_inds)+'-father')

                                # Likely need to exclude these from subsequent zygotic modification
                                elif mod[0] == 'zygotic' and mod[2] in mod_sprin_a_l:
                                    mod_a_ind = mod_sprin_a_l.index(mod[2])
                                    if (all_option(mod[1][0], mother_alleles) and 
                                        all_option(mod[1][1], father_alleles) and
                                        all_option(mod[1][2], mod_sprin_a_l)):
                                        if mod_a_ind%2 == 0 and mod_sprin_a_l[mod_a_ind+1] == mod[2]:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][-1]
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)
                                                
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)
                                                
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)

                                        elif mod_a_ind%2 == 0:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)

                                        elif mod_a_ind%2 == 1:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)

                                # Likely need to exclude these from subsequent somatic modification
                                elif mod[0] == 'somatic' and mod[2] in mod_sprin_a_l:
                                    mod_a_ind = mod_sprin_a_l.index(mod[2])
                                    if all_option(mod[1], mod_sprin_a_l):
                                        if mod_a_ind%2 == 0 and mod_sprin_a_l[mod_a_ind+1] == mod[2]:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][-1]
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)
                                                
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)
                                                
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)

                                        elif mod_a_ind%2 == 0:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][0] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]-1] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)

                                        elif mod_a_ind%2 == 1:
                                            mod_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][-1]
                                            for sub_mod in mod[3]:
                                                temp_sprin = copy.deepcopy(mod_sprin)
                                                temp_sprin[0][int(mod_a_ind/2)][1] = sub_mod
                                                temp_sprin[1][mod_dict[(mod[0],mod[2])][1]] = d_a[mod_i][mod[3].index(sub_mod)]
                                                mod_sprin_list.append(temp_sprin)
#                     print(mod_sprin_list)
                    for mod_sprin in mod_sprin_list:
#                         print(mod_sprin)
#                         print(mod_sprin[1])
#                         print(np.prod(mod_sprin[1]))
                        offspring_modified[0].append(mod_sprin[0])
                        offspring_modified[1].append(np.prod(mod_sprin[1]))
                cross_dict[(genotypes[1].index(mother), genotypes[2].index(father))] = offspring_modified
#             print(offspring_modified)
            n_sprin = np.random.poisson(n_o)
            new_adult_inds = np.random.choice(range(len(offspring_modified[0])), n_sprin, offspring_modified[1])
            new_adults = [offspring_modified[0][n_a_i] for n_a_i in new_adult_inds]

            survival_modifier = g_f/(1+(g_f-1)*(len(adults[0][gen])+len(adults[1][gen]))/K)

            # This will be adjusted with varying sex_mods
            for new_adult in new_adults:
                new_adult_a = allele_list(new_adult, 'genotype')

                if sex_det[2] != []:
                    if any([all_option(s_c, new_adult_a) for s_c in sex_det[2]]):
                        if np.random.binomial(1, 2/n_o*(1-FC[1][genotypes[1].index(new_adult)])*survival_modifier):
                            adults[1][gen+1].append(new_adult)
                        
                elif sex_det[1] != []:
                    if any([all_option(s_c, new_adult_a) for s_c in sex_det[1]]):
                        if np.random.binomial(1, 2/n_o*(1-FC[0][genotypes[1].index(new_adult)])*survival_modifier):
                            adults[1][gen+1].append(new_adult)
                        
                elif 'Y' in new_adult_a:
                    if np.random.binomial(1, 2/n_o*(1-FC[1][genotypes[1].index(new_adult)])*survival_modifier):
                        adults[1][gen+1].append(new_adult)

                else:
#                     if np.random.binomial(1, 2/n_o*(1-FC[0][genotyypes[0].index(new_adult)])*survival_modifier):
#                         adults[0][gen+1].append(new_adult)
                    
#                     print(FC[0][genotypes[0].index(new_adult)])
#                     print(survival_modifier)
    
                    if np.random.binomial(1, 2/n_o*(1-FC[0][genotypes[0].index(new_adult)])*survival_modifier):
                        if np.random.binomial(1, 0.5):
                            adults[0][gen+1].append(new_adult)
                        else:
                            adults[1][gen+1].append(new_adult)
    
    return(adults, cross_dict, genotypes)
        


# In[567]:


def drive_plotter(frequencies, genotypes, plot_values, plot_gens = 300, 
                  plot_type = 'g_freq', plot_sexed = 0):
    """plots drive behavior based on specifid genotypes or alleles of interest of specified number of generations"""
    
    geno_alleles = genotypes[4:]
    
    if plot_type == 'g_freq':
        if plot_sexed == 0:
            plot_freqs = np.zeros([len(frequencies[0][:,1]), len(plot_values)])
            for a_i, allele in enumerate(plot_values):
                freqs = np.zeros(len(frequencies[0][:,1]))
                for sex in [0, 1]:
                    for g_i, genotype in enumerate(geno_alleles[sex]):
                        if allele in genotype:
                            freqs += frequencies[sex][:, g_i]
#                 print(freqs)
                plot_freqs[:, a_i] = freqs
                            
            
        elif plot_sexed == 1:
            plot_freqs = np.zeros([plot_gens+1, len(plot_values*2)])
            
        else:
            return('Throw an error here')
        
    elif plot_type == 'a_freq':
        if plot_sexed == 0:
            plot_freqs = np.zeros([len(frequencies[0][:,1]), len(plot_values)])
            for a_i, allele in enumerate(plot_values):
                freqs = np.zeros(len(frequencies[0][:,1]))
                for sex in [0, 1]:
                    for g_i, genotype in enumerate(geno_alleles[sex]):
                        if allele in genotype:
                            if all_option([allele, allele], genotype):
                                freqs += frequencies[sex][:, g_i]
                            else:
                                freqs += frequencies[sex][:, g_i] * 1/2
                            
                plot_freqs[:, a_i] = freqs
                
        elif plot_sexed == 1:
            plot_freqs = np.zeros([plot_gens+1, len(plot_values*2)])
       
    elif plot_type == 'genotypes':
        if plot_sexed == 0:
            plot_vals = np.zeros([len(frequencies[0][:,1]), len(plot_values)])
            for a_i, allele in enumerate(plot_values):
                freqs = np.zeros(len(frequencies[0][:,1]))
                for sex in [0, 1]:
                    for g_i, genotype in enumerate(geno_alleles[sex]):
                        if allele in genotype:
                            freqs += frequencies[sex][:, g_i]
#                 print(freqs)
                plot_freqs[:, a_i] = freqs
                            
            
        elif plot_sexed == 1:
            plot_freqs = np.zeros([plot_gens+1, len(plot_values*2)])
            
        else:
            return('Throw an error here')
        
    else:
        return('Throw an error here')
    
    plt.figure(figsize = (10,10))
    plt.plot(plot_freqs, linewidth=7.0)
    #     plt.yscale('linear')
    plt.legend(all_alleles)
    plt.axis([0, plot_gens, 0, 1])
    plt.title('Phenotype Frequency')
    plt.grid(True)
#     return(plot_freqs)


# In[ ]:


r_d = [0.2, 0.3, 0.4]
r_d = [[0.2, 0.3, 0.4], [0.4, 0.2, 0.1]]
n_r_d = 3

if isinstance(r_d[0], float) and len(r_d) == n_r_d:
    r_d_temp = r_d
    r_d = [r_d_temp, r_d_temp]

elif isinstance(r_d[0], list):
    if len(r_d[0]) == len(r_d[1]) == n_r_d:
        err = 'NA'
    else:
        err = 'throw error: must have the same number of recombination rates for each sex'

else:
    err = 'throw error: r_d must be either a single list or a pair of lists of     length equal to the number of recombination distances'

r_d_full = []
for r_ind, rec in enumerate(r_d):
    r_d_temp = []
    [r_d_temp.append([r, 1-r]) for r in rec]
    r_d_full.append(r_d_temp)

for rec in r_d_full[0:2]:
    r_d_full.append(list(product(*rec)))

for rec in r_d_full[2:4]:
    r_d_full.append([np.prod(rd) for rd in rec])
    
r_d_full.append(list(product(*r_d_full[4:6])))
r_d_full.append([np.prod(rd) for rd in r_d_full[6]])

r_d_full[7]


# In[20]:


"Test case for a two locus ClvR scenario, when everything is working this is a first simple test"
alleles = [['T', 'A'], ['Xc', 'X', 'Y']]
mods = [['germline', ['T'], 'X', ['Xc']],
        ['zygotic', [['T'], [], []], 'X', ['Xc']]
       ]
sex_det = ['XY', [], []]


# In[378]:


"Test case for a two locus ClvR scenario, when everything is working this is a first simple test"
alleles = [['T', 'C', 'A'], ['X', 'Y']]
mods = [['germline', ['T'], 'A', ['T', 'C']],
        ['zygotic', [['T'], [], []], 'A', ['T', 'C']]
       ]
sex_det = ['XY', [], []]


# In[955]:


"Test case for a 'one locus' ClvR scenario"
alleles = [['T', 'C', 'A']]
mods = [['germline', ['T'], 'A', ['T', 'C']],
        ['zygotic', [['T'], [], []], 'A', ['C']]
       ]
sex_det = ['autosomal', [], []]


# In[1067]:


num_gens = 20
# d_a = [[0.95], [0.85], [0.9]]
# d_a = [[0, 0], [0, 0], [0]]
# d_a = [[0.5, 0.5], [0.5, 0.5], [0]]
# d_a = [[0.5, 0], [0.5, 0], [0]]
# d_a = [[0, 0.5], [0, 0.5], [0]]
# d_a = [[0.75, 0.25], [0.75, 0.25], [0]]
intro = [[0.5], [0]]
d_a = [[0, 0.25], [0, 0.25], [1]]
r_d = [0]
# [0, 0, 0, 0, 0], 
# [0.2, 0.25, 0, 0, 0], 
f_c = [['dominant', 'dominant', 'dominant', 'dominant', 'dominant'], 
       [0.2, 0.25, 0, 0, 0], 
       [['C', 'C'],], 
       [[['T']], ]]

adults, cross_dict, genotypes = stochastic_sim(alleles, mods, sex_det, num_gens, intro, d_a, r_d, f_c, 
                                    K = 10000, n_o = 100, g_f = 10, cross_dict = {})


# In[1056]:


print(str(adults[1][200]))


# In[1061]:


genotypes[1][8]


# In[1068]:


cross_dict


# In[1064]:


cross_dict


# In[832]:


c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict = cross_matrix_generator(alleles, mods, sex_det)


# In[833]:


num_gens = 300
# d_a = [[0.95], [0.85], [0.9]]
# d_a = [[0, 0], [0, 0], [0]]
# d_a = [[0.5, 0.5], [0.5, 0.5], [0]]
# d_a = [[0.5, 0], [0.5, 0], [0]]
# d_a = [[0, 0.5], [0, 0.5], [0]]
# d_a = [[0.75, 0.25], [0.75, 0.25], [0]]
d_a = [[0, 1], [0, 1], [1]]
r_d = [0]
# [0, 0, 0, 0, 0], 
# [0.2, 0.25, 0, 0, 0], 
f_c = [['dominant', 'dominant', 'dominant', 'dominant', 'dominant'], 
       [0.2, 0.25, 0, 0, 0], 
       [['C', 'C'],], 
       [[['T']], ]]

frequencies = drive_simulation(c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict, 
                               num_gens, d_a, r_d, f_c, intro = [[0.5], [0]], 
                               sim_type = 'deterministic', c_m_type = 'matrix', 
                               m_r = 0.01, a_r = [0, [0.3], [0], 200, 50])


# In[834]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20, plot_type = 'g_freq')


# In[835]:


# frequencies[0][:,2]+frequencies[0][:,6]
frequencies[0][:,0]


# In[910]:


"Test case for a two locus ClvR scenario"
alleles = [['T', 'W'], ['C', 'A']]
mods = [['germline', ['T'], 'A', ['C']],
        ['zygotic', [['T'], [], []], 'A', ['C']]
       ]
sex_det = ['autosomal', [], []]


# In[903]:


"Test case for a two locus ClvR scenario"
alleles = [['T', 'W'], ['C', 'A']]
mods = []
sex_det = ['autosomal', [], []]


# In[911]:


c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict = cross_matrix_generator(alleles, mods, sex_det)


# In[915]:


num_gens = 300
d_a = [[1], [1], [1]]
# d_a = []
r_d = [50]
# [0, 0, 0, 0, 0], 
# [0.2, 0.25, 0, 0, 0], 
f_c = [['dominant', 'dominant', 'dominant', 'dominant', 'dominant'], 
       [0.2, 0, 0.25, 0, 0], 
       [['C', 'C'],], 
       [[['T']], ]]

frequencies = drive_simulation(c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict, 
                               num_gens, d_a, r_d, f_c, intro = [[0.5], [0]], 
                               sim_type = 'deterministic', c_m_type = 'matrix', 
                               m_r = 0.01, a_r = [0, [0.3], [0], 200, 50])


# In[916]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20, plot_type = 'g_freq')


# In[899]:


# frequencies[0][:,2]+frequencies[0][:,6]
frequencies[0][:,0]


# In[918]:


"Test case for a two locus ClvR scenario"
alleles = [['T', 'W'], ['S', 'A'], ['C', 'U']]
mods = [['germline', ['T'], 'U', ['C']],
        ['zygotic', [['T'], [], []], 'U', ['C']]
       ]
sex_det = ['autosomal', [], []]


# In[919]:


c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict = cross_matrix_generator(alleles, mods, sex_det)


# In[920]:


num_gens = 300
d_a = [[1], [1], [1]]
# d_a = []
r_d = [0, 50]
# [0, 0, 0, 0, 0], 
# [0.2, 0.25, 0, 0, 0], 
f_c = [['dominant', 'dominant', 'dominant', 'dominant', 'dominant', 'dominant'], 
       [0.1, 0, 0.1, 0, 0.25, 0], 
       [['C', 'C'],], 
       [[['T']], ]]

frequencies = drive_simulation(c_m, genotypes, all_alleles, num_loci, n_r_d, l_r_d_d, mod_dict, 
                               num_gens, d_a, r_d, f_c, intro = [[0.5], [0]], 
                               sim_type = 'deterministic', c_m_type = 'matrix', 
                               m_r = 0.01, a_r = [0, [0.3], [0], 200, 50])


# In[921]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20, plot_type = 'g_freq')


# In[922]:


# frequencies[0][:,2]+frequencies[0][:,6]
frequencies[0][:,0]


# In[870]:


x == frequencies[0]


# In[696]:


c_m_test = frequencies[2]


# In[650]:


c_m_test[0][2,2,:]


# In[645]:


c_m_test[0][2,2,:]


# In[656]:


c_m_test[0][2,2,:]


# In[697]:


c_m_test[0][2,2,:]


# In[663]:


genotypes[4]


# In[615]:


# d_a = [[0, 0], [0, 0], [0, 0]]
c_m_test[0][:, 0, :]


# In[534]:


# d_a = [[0.5, 0.5], [0.5, 0.5], [0, 0]]
c_m_test[0][:, 0, :]


# In[537]:


# d_a = [[0.75, 0.25], [0.75, 0.25], [0, 0]]
c_m_test[0][:, 0, :]


# In[540]:


# d_a = [[1, 0], [1, 0], [0, 0]]
c_m_test[0][:, 0, :]


# In[233]:


genotypes[4]


# In[234]:


genotypes[5]


# In[209]:


females = frequencies[0]
# males = frequencies[1]
males = frequencies[1]

# sim = pd.DataFrame(columns=['Generation', 'Sex', 'Genotype', 'Frequency'])
# for sex in [0,1]:
#     for geno_ind, genotype in enumerate(genotypes[sex+4]):
#         if sex == 0:
#             df_dict = {'Generation': [int(x) for x in np.linspace(0, num_gens, num = num_gens+1)], 
#                        'Sex': ['Female']*(num_gens+1), 
#                        'Genotype': ["".join(genotype)]*(num_gens+1), 
#                        'Frequency': females[:, geno_ind]}
#         elif sex == 1:
#             df_dict ={'Generation': [int(x) for x in np.linspace(0, num_gens, num = num_gens+1)], 
#                       'Sex': ['Male']*(num_gens+1), 
#                       'Genotype': ["".join(genotype)]*(num_gens+1), 
#                       'Frequency': males[:, geno_ind]}
        
#         df_temp = pd.DataFrame(data = df_dict)
#         sim = sim.append(df_temp, ignore_index=True)

sim_females = pd.DataFrame(females, columns = ["".join(x) for x in genotypes[4]])
sim_males = pd.DataFrame(males, columns = ["".join(x) for x in genotypes[5]])
sim2 = pd.concat([sim_females, sim_males], axis = 1)
sim2.index.names = ['Generation']
sim = pd.DataFrame()
for allele in all_alleles:
    sim[allele+'-allele_freq'] = (np.sum(sim2.filter(regex=allele), axis = 1) 
                                  + np.sum(sim2.filter(regex=allele*2), axis = 1))/2
    sim[allele+'-genotype_freq'] = np.sum(sim2.filter(regex=allele), axis = 1)
sim3 = pd.DataFrame()
sim3['Locus 1 allele frequency'] = (sim['T-allele_freq'].values 
                                    + sim['C-allele_freq'].values 
                                    + sim['A-allele_freq'].values)

sim3['Locus 2 allele frequency'] = (sim['X-allele_freq'].values 
                                    + sim['Y-allele_freq'].values)

sim3['Locus 1 genotype frequency'] = (sim['T-genotype_freq'].values 
                                      + sim['C-genotype_freq'].values 
                                      + sim['A-genotype_freq'].values)

sim3['Locus 2 genotype frequency'] = (sim['X-genotype_freq'].values 
                                      + sim['Y-genotype_freq'].values)

sim3


# In[203]:


print(np.sum(females[1]))
print(np.sum(males[1]))


# In[207]:


np.sum([0.03239063, 0.00809766, 0.05693666, 0.03239063, 0.01012207, 0.07211976,
 0.09109865, 0.02884791, 0.21635929])


# In[191]:


np.sum([8.097658e-03, 2.882766e-02, 5.693666e-02, 2.591250e-03, 0.009312, 2.307832e-02, 2.277466e-02, 8.365893e-02, 2.163593e-01, 7.710843e-02, 0.000000e+00, 1.084337e-01, 0.000000e+00, 0.000000, 0.000000e+00, 1.084337e-01, 0.000000e+00, 2.060241e-01])


# In[107]:


sim.loc[(sim['Sex'] == 'Female') & ('sim['Genotype'] ), :]


# In[114]:


sim['Genotype']


# In[553]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20, plot_type = 'a_freq')


# In[549]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20)


# In[67]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20)


# In[27]:


drive_plotter(frequencies, genotypes, plot_values = all_alleles, plot_gens = 20, plot_type = 'a_freq')


# In[ ]:





# In[ ]:


p1 = drive_plotter(frequencies, genotypes, plot_gens = 4)
p1


# In[ ]:


p2 = drive_plotter(frequencies, genotypes, plot_gens = 4, plot_type = 'a_freq')
p2


# In[ ]:


def add_new_data_to_dataframe(df, data, run, freq_type, pop, num_gens, all_alleles):
    """
    add_new_data_to_dataframe takes new data and combines it with the 
    previous data into a single dataframe.
    """
    
    df_temp = pd.DataFrame(data, columns = all_alleles)
    df_temp = df_temp.melt()
    df_temp['Generation'] = [int(gen) for gen in np.linspace(0, num_gens, num_gens+1)]*len(all_alleles)
    df_temp = df_temp.rename(columns={"variable": "Allele", "value": "Frequency"})
    df_temp['Frequency Type'] = freq_type
    df_temp['Run'] = run
    df_temp['Population'] = pop
    
    df = df.append(df_temp)
    
    return(df)

def multi_run_plots(df, gene_drive_name, num_pops, plot_alleles, leg_pos, fig_width, fig_height):
    """
    multi_run_plots handles plotting for the multi_run function.
    """
    
    plot_list = []
    
    for allele in plot_alleles:
        for pop in range(1, num_pops+1):
            
            plot = hv.Points(
                data=df[(df['Allele'] == allele) & (df['Frequency Type'] == 'phenotype') & (df['Population'] == pop)],
                kdims=['Generation', 'Frequency'],
                vdims=['Run'],
            ).groupby(
                'Run'
            ).opts(
                title=str(gene_drive_name) + ' Frequency of ' + str(allele) + ' in population ' + str(pop),
                tools=['hover'],
                legend_position=leg_pos,
                width=fig_width,
                height=fig_height
            ).overlay(
            )

            plot_list.append(plot)
        
    data_plot = plot_list[0]

    if len(plot_list) >= 2:
        for p in range(1, len(plot_list)):
            data_plot += plot_list[p]

    return(data_plot)


# In[ ]:


x = np.zeros([20,2])
x[:,1]


# In[ ]:


dimensional_factors = np.array([[[1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8]],
                                [[1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8]],
                                [[1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8]],
                                [[1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8]]])
dimensional_factors.shape
np.sum(dimensional_factors, axis = (1,2))
# dimensional_factors


# In[ ]:


all_alleles[0]


# In[ ]:


females


# In[ ]:


np.array([1, 2])


# In[ ]:


males


# In[ ]:


np.dot(females,males)


# In[ ]:


np.dot(np.array([1, 2, 3]), np.array([3, 4]))


# In[ ]:


# alleles = [['F', 'S', 'W'],['L', 'O'],['M', 'N']]

# mods = [['F', 'O', 'L', 'NA'], ['F', 'O', 'L', 'carryover'], ['S', 'N', 'M', 'NA'], ['S', 'N', 'M', 'carryover']]


# In[ ]:


# 1L ClvR XY
alleles = [['C', 'W'],['Xc', 'X', 'Y']]

mods = [['C', 'X', 'Xc', 'NA'], ['C', 'X', 'Xc', 'carryover']]


# In[ ]:


# 1L ClvR
alleles = [['T', 'W'],['C', 'A']]

mods = [['T', 'A', 'C', 'NA'], ['T', 'A', 'C', 'carryover']]


# In[ ]:


# 2L ClvR XY
alleles = [['Clv', 'W1'],['R', 'W2'],['Xc', 'X', 'Y']]

mods = [[['Clv', 'R'], 'X', 'Xc', 'NA'], [['Clv', 'R'], 'X', 'Xc', 'carryover']]


# In[ ]:


# 1L ClvR in female fertility gene
alleles = [['T', 'W'],['C', 'A']]

mods = [['T', 'A', 'C', 'NA'], ['T', 'A', 'C', 'carryover']]


# In[ ]:


## Generate Plot Frequencies ##
# M_g_l = Male_genotypes_lists, 
# F_g_l = Female_genotypes_lists, 

M_g_l = []
F_g_l = []

for g_i, genotype in enumerate(M_g):
    new_M_g_l = AlleleList(genotype, 'genotype')
    M_g_l.append(new_M_g_l)

for g_i, genotype in enumerate(F_g):
    new_F_g_l = AlleleList(genotype, 'genotype')
    F_g_l.append(new_F_g_l)

# M_a = Male_alleles, 
# F_a = Female_alleles, 
# all_a = all_alleles, 
    
M_a = []
F_a = []
all_a = []

for locus in alleles:
    for allele in locus:
        all_a.append(allele)
        
        for m_g_l in M_g_l:
            if allele in m_g_l:
                M_a.append(allele)
                break

        for f_g_l in F_g_l:
            if allele in f_g_l:
                F_a.append(allele)
                break
                
# M_a_f_s = Male_allele_frequency_s,
# F_a_f_s = Female_allele_frequency_s,
# all_a_f_s = all_allele_frequency_s,
# M_p_f_s = Male_phenotype_frequency_s,
# F_p_f_s = Female_phenotype_frequency_s,
# all_p_f_s = all_phenotype_frequency_s,

M_a_f_s = []
F_a_f_s = []
all_a_f_s = []
M_p_f_s = []
F_p_f_s = []
all_p_f_s = []

for a in range(len(M_a)):
    M_a_f_s += ' '
    M_p_f_s += ' '
for a in range(len(F_a)):
    F_a_f_s += ' '
    F_p_f_s += ' '
for a in range(len(all_a)):
    all_a_f_s += ' '
    all_p_f_s += ' '

for g_l_i, g_l in enumerate(M_g_l):
    for a_i, allele in enumerate(g_l):
        if a_i != len(g_l)-1 and g_l[a_i+1] == allele:
            all_a_f_s[all_a.index(allele)] += 'M[g,' + str(g_l_i) + ']/2+'
            M_a_f_s[M_a.index(allele)] += 'M[g,' + str(g_l_i) + ']/2+'
        else:
            all_a_f_s[all_a.index(allele)] += 'M[g,' + str(g_l_i) + ']/2+'
            all_p_f_s[all_a.index(allele)] += 'M[g,' + str(g_l_i) + ']+'
            M_a_f_s[M_a.index(allele)] += 'M[g,' + str(g_l_i) + ']/2+'
            M_p_f_s[M_a.index(allele)] += 'M[g,' + str(g_l_i) + ']+'

for g_l_i, g_l in enumerate(F_g_l):
    for a_i, allele in enumerate(g_l):
        if a_i != len(g_l)-1 and g_l[a_i+1] == allele:
            all_a_f_s[all_a.index(allele)] += 'F[g,' + str(g_l_i) + ']/2+'
            F_a_f_s[F_a.index(allele)] += 'F[g,' + str(g_l_i) + ']/2+'
        else:
            all_a_f_s[all_a.index(allele)] += 'F[g,' + str(g_l_i) + ']/2+'
            all_p_f_s[all_a.index(allele)] += 'F[g,' + str(g_l_i) + ']+'
            F_a_f_s[F_a.index(allele)] += 'F[g,' + str(g_l_i) + ']/2+'
            F_p_f_s[F_a.index(allele)] += 'F[g,' + str(g_l_i) + ']+'

for f_i, frequency in enumerate(all_p_f_s):
    all_p_f_s[f_i] = frequency[1:-1]
for f_i, frequency in enumerate(all_a_f_s):
    all_a_f_s[f_i] = frequency[1:-1]

for f_i, frequency in enumerate(M_p_f_s):
    M_p_f_s[f_i] = frequency[1:-1]
for f_i, frequency in enumerate(M_a_f_s):
    M_a_f_s[f_i] = frequency[1:-1]

for f_i, frequency in enumerate(F_p_f_s):
    F_p_f_s[f_i] = frequency[1:-1]
for f_i, frequency in enumerate(F_a_f_s):
    F_a_f_s[f_i] = frequency[1:-1]


# In[ ]:


F[0]


# In[ ]:





# In[ ]:


# Simulate population
for gen in range(1,numGens+1):
    
    males[gen] = []
    females[gen] = []
    
    for mo_g_i in females[gen-1]:
        
        new_males = []
        new_females = []
    
        mother = F_g[mo_g_i]
        fa_g_i = random.choice(males[gen-1])
        father = M_g[fa_g_i]

        if M_cm.ndim == 3:

            F_cm_weights[fa_g_i][mo_g_i] = [eval(daughter) for daughter in d_p]
            M_cm_weights[fa_g_i][mo_g_i] = [eval(son) for son in s_p]
            
        else:
            
            
        
        offspring = np.random.poisson(fertility*(1-Ffc[mo_g_i])*(1-Mfc[fa_g_i]))
        daughters = np.random.binomial(offspring, 0.5)
        sons = offspring-daughters
        
        new_females.extend([daughter for daughter in random.choices(F_g, weights = F_cm_weights[fa_g_i][mo_g_i], k=daughters)])
        new_males.extend([son for son in random.choices(M_g, weights = M_cm_weights[fa_g_i][mo_g_i], k=sons)])
        
        female_survivors = [np.random.binomial(1, (1/fertility)*(Eq/(len(males[gen-1])+len(females[gen-1])))*(1-Ffc[x])) for x in new_females]
        male_survivors = [np.random.binomial(1, (1/fertility)*(Eq/(len(males[gen-1])+len(females[gen-1])))*(1-Mfc[x])) for x in new_males]
        
        females[gen].extend([new_females[x] for x in range(len(new_females)) if female_survivors[x] != 0])
        males[gen].extend([new_males[x] for x in range(len(new_males)) if male_survivors[x] != 0])
        
    for genotype in range(n_F_g):
        F[gen, genotype] = females[gen].count(genotype)
        
    for genotype in range(n_M_g):
        M[gen, genotype] = males[gen].count(genotype)
    
    


# In[ ]:


len(M_g)


# In[ ]:


len(males[9])


# In[ ]:





# In[ ]:


## Plot Curves ##

plot_gens = 100
M_a_f = np.zeros([numGens+1,len(M_a_f_s)])
F_a_f = np.zeros([numGens+1,len(F_a_f_s)])
all_a_f = np.zeros([numGens+1,len(all_a_f_s)])
M_p_f = np.zeros([numGens+1,len(M_p_f_s)])
F_p_f = np.zeros([numGens+1,len(F_p_f_s)])
all_p_f = np.zeros([numGens+1,len(all_p_f_s)])
M_wt_f = np.zeros(numGens+1)
F_wt_f = np.zeros(numGens+1)

for f_i, frequency in enumerate(M_a_f_s):
    for g in range(numGens+1):
        M_a_f[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(M_p_f_s):
    for g in range(numGens+1):
        M_p_f[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(F_a_f_s):
    for g in range(numGens+1):
        F_a_f[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(F_p_f_s):
    for g in range(numGens+1):
        F_p_f[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(all_a_f_s):
    for g in range(numGens+1):
        all_a_f[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(all_p_f_s):
    for g in range(numGens+1):
        all_p_f[g][f_i] = eval(frequency)
    


if XY == 1:
    # Phenotype Frequencies
    plt.subplot(221)
    plt.plot(all_p_f)
    #     plt.yscale('linear')
    plt.legend(all_a)
    plt.axis([0, plot_gens, 0, Eq*1.5])
    plt.title('Phenotype Frequency')
    plt.grid(True)

    # Allele Frequencies
    plt.subplot(222)
    plt.plot(all_a_f)
    #     plt.yscale('linear')
    # plt.legend(all_a)
    plt.axis([0, plot_gens, 0, Eq*1.5])
    plt.title('Allele Frequency')
    plt.grid(True)

    # Gendered Phenotype Frequencies
    plt.subplot(223)
    plt.plot(M_p_f)
    plt.plot(F_p_f)
    #     plt.yscale('linear')
    # plt.legend(M_a + F_a)
    plt.axis([0, plot_gens, 0, Eq*1.5])
    # plt.title('Gendered Phenotype Frequency')
    plt.grid(True)

    # Gendered Allele Frequencies
    plt.subplot(224)
    plt.plot(M_a_f)
    plt.plot(F_a_f)
    #     plt.yscale('linear')
    plt.legend(M_a + F_a)
    plt.axis([0, plot_gens, 0, Eq*1.5])
    # plt.title('Gendered Allele Frequency')
    plt.grid(True)
    
else:
    # Phenotype Frequencies
    plt.figure(figsize = (10,10))
    plt.plot(all_p_f, linewidth=7.0)
    #     plt.yscale('linear')
    plt.legend(all_a)
    plt.axis([0, plot_gens, 0, Eq*1.5])
    plt.title('Phenotype Frequency')
    plt.grid(True)


# In[ ]:


# Genotype Frequencies
plt.figure(figsize = (10,10))
plt.plot(all_p_f[:,[0, 2]], linewidth=7.0)
#     plt.yscale('linear')
plt.legend(['ClvR', 'Cleaved non-ClvR locus'])
plt.axis([0, numGens, 0, Eq*1.5])
plt.title('Genotype Frequency')
plt.grid(True)


# In[ ]:


# Allele Frequencies
plt.figure(figsize = (10,10))
plt.plot(all_a_f[:,[0, 2]], linewidth=7.0)
plt.legend(['ClvR', 'Cleaved non-ClvR locus'])
plt.axis([0, 600, 0, 1])
plt.title('Allele Frequency')
plt.grid(True)


# In[ ]:


M_g


# In[ ]:


F[60]


# In[ ]:


for x in all_a_f[0:40, :]:
    print(x)


# In[ ]:


for x in all_a_f[60, :]:
    print(x)


# In[ ]:


all_a


# In[ ]:


for x in all_p_f[:, 0]:
    print(x)


# In[ ]:


# 2L ClvR XY
lethal = [['Xc', 'Xc'], ['Xc', 'Y']]
rescue = [[['R']], [['R']]]

alleles_m_fc = [0.05, 0, 0.05, 0, 0, 0, 0]
alleles_f_fc = alleles_m_fc

numGens = 600
IF = 0.5
# IF_genotype = [21]
IF_genotype = [21, 42]
IF2 = 0
M1 = 0
F1 = 0
Mfc = fitness_cost(alleles_m_fc, drive_Fitness(lethal, rescue, M_g_l), M_g_l, M_a, 'additive')
Ffc = fitness_cost(alleles_f_fc, drive_Fitness(lethal, rescue, F_g_l), F_g_l, F_a, 'additive')
ar = 10
ar_g = [21, 42]
# ar_g = [-1]
ar_f = 0.5
ar_gen_freq = 30
M_li_b = [0, 0]
F_li_b = [0, 0]
D_A = [1, 1]
mig_rate = 0.01
pop_2_rel_size = 1


# In[ ]:


if M_li_b == 0:
    M_li = np.ones(num_loci-1)*0.5
    F_li = np.ones(num_loci-1)*0.5
else:
    M_li = np.array(M_li_b)*0.5+0.5
    F_li = np.array(F_li_b)*0.5+0.5

M_un = 1-M_li
F_un = 1-F_li

for fc_i, fc in enumerate(Mfc):
    if fc > 1:
        Mfc[fc_i] = 1
        print('Check fitness costs, this shouldn\'t be printing anything')
        
for fc_i, fc in enumerate(Ffc):
    if fc > 1:
        Ffc[fc_i] = 1
        print('Check fitness costs, this shouldn\'t be printing anything')



D_A_M = []  # Drive Activity for Males (mxn table, m = number of genotypes, n = number of drive activities)
D_A_F = []

D_A_M_r = [] # Drive Activity for Males rates (1xn list, n = drive activity rates for Males)
D_A_F_r = []

for activity in range(D_A_tc):
    D_A_M.append([])
    D_A_F.append([])
    
    D_A_M_r.append([])
    D_A_F_r.append([])
    
    for male in range(n_M_g):
        D_A_M[activity].append([])
        
    for female in range(n_F_g):
        D_A_F[activity].append([])

for activity in range(D_A_tc):
    if activity == 1:
        D_A_M_r[activity] = 0
        D_A_F_r[activity] = eval(D_A_F_r_s[activity])
        
    elif activity == 3:
        D_A_M_r[activity] = 0
        D_A_F_r[activity] = eval(D_A_F_r_s[activity])
        
    else:
        
        D_A_M_r[activity] = eval(D_A_M_r_s[activity])
        D_A_F_r[activity] = eval(D_A_F_r_s[activity])
    
    for male in range(n_M_g):
        D_A_M[activity][male] = eval(D_A_M_s[activity][male])
        
    for female in range(n_F_g):
        D_A_F[activity][female] = eval(D_A_F_s[activity][female])


# Convert Cross Matrices to numerical form
M_cm_cv = np.zeros([n_M_g, n_M_g, n_F_g])
F_cm_cv = np.zeros([n_F_g, n_M_g, n_F_g])

for index1, genotype in enumerate(M_cm):
    for index2, mother in enumerate(genotype):
        for index3, father in enumerate(mother):
            if isinstance(M_cm[index1][index2][index3], str):
                M_cm_cv[index1][index2][index3] = eval(M_cm[index1][index2][index3])
            elif M_cm[index1][index2][index3] == None:
                M_cm_cv[index1][index2][index3] = 0
            else:
                print('Inappropriate Cross Matrix entry')

for index1, genotype in enumerate(F_cm):
    for index2, mother in enumerate(genotype):
        for index3, father in enumerate(mother):
            if isinstance(F_cm[index1][index2][index3], str):
                F_cm_cv[index1][index2][index3] = eval(F_cm[index1][index2][index3])
            elif F_cm[index1][index2][index3] == None:
                F_cm_cv[index1][index2][index3] = 0
            else:
                print('Inappropriate Cross Matrix entry')



# Initialize male and female population proportions
M_1 = np.matrix(np.zeros((numGens+1, n_M_g)))
F_1 = np.matrix(np.zeros((numGens+1, n_F_g)))

M_2 = np.matrix(np.zeros((numGens+1, n_M_g)))
F_2 = np.matrix(np.zeros((numGens+1, n_F_g)))

# Set initial population proportions
if len(IF_genotype) == 1:
    if M1 == 0:
        M1 = np.zeros(n_M_g)
        M1[IF_genotype] = IF/2
        M1[-1] = (1-IF)/2
        
        M2 = np.zeros(n_M_g)
        M2[-1] = 1/2

    if F1 == 0:
        F1 = np.zeros(n_F_g)
        F1[-1] = 1/2
        
        F2 = np.zeros(n_F_g)
        F2[-1] = 1/2
else:
    if M1 == 0:
        M1 = np.zeros(n_M_g)
        M1[IF_genotype[0]] = IF/2
        M1[-1] = (1-IF)/2
        
        M2 = np.zeros(n_M_g)
        M2[-1] = 1/2

    if F1 == 0:
        F1 = np.zeros(n_F_g)
        F1[IF_genotype[1]] = IF/2
        F1[-1] = (1-IF)/2
        
        F2 = np.zeros(n_F_g)
        F2[-1] = 1/2
    

M_1[0,:] = M1
F_1[0,:] = F1

M_2[0,:] = M2
F_2[0,:] = F2


# Initialize temporary current population frequencies
M_1_temp = np.array(np.zeros(n_M_g))

F_1_temp = np.array(np.zeros(n_F_g))

M_2_temp = np.array(np.zeros(n_M_g))

F_2_temp = np.array(np.zeros(n_F_g))


# In[ ]:


# Simulate population
ar_copy = ar
sigma_1 = np.zeros(numGens+1)
sigma_2 = np.zeros(numGens+1)
for gen in range(1,numGens+1):


    # Generate proportion of each genotype pairing
    presentcross_1 = np.transpose(M_1[gen-1])*F_1[gen-1]
    
    presentcross_2 = np.transpose(M_2[gen-1])*F_2[gen-1]


    # Generate gross proportions for each genotype
    for M_geno in range(n_M_g):
        M_1_temp[M_geno] = np.sum(np.multiply(M_cm_cv[M_geno][:][:],presentcross_1))
        
        M_2_temp[M_geno] = np.sum(np.multiply(M_cm_cv[M_geno][:][:],presentcross_2))

    for F_geno in range(n_F_g):
        F_1_temp[F_geno] = np.sum(np.multiply(F_cm_cv[F_geno][:][:],presentcross_1))
        
        F_2_temp[F_geno] = np.sum(np.multiply(F_cm_cv[F_geno][:][:],presentcross_2))
        
    sigma_1[gen] = np.sum(np.multiply(M_1_temp, np.subtract([1]*len(Mfc), Mfc))) + np.sum(np.multiply(F_1_temp, np.subtract([1]*len(Ffc), Ffc)))
    
    sigma_2[gen] = np.sum(np.multiply(M_2_temp, np.subtract([1]*len(Mfc), Mfc))) + np.sum(np.multiply(F_2_temp, np.subtract([1]*len(Ffc), Ffc)))
    
#     if gen%ar_gen_freq == False and ar_copy > 0:
#         M[gen][:] = np.multiply(np.multiply(M_temp, np.subtract([1]*len(Mfc), Mfc))/sigma[gen], 1-ar_f)
#         F[gen][:] = np.multiply(np.multiply(F_temp, np.subtract([1]*len(Ffc), Ffc))/sigma[gen], 1-ar_f)
#         M[gen, ar_g] += ar_f/2
#         F[gen,ar_g] += ar_f/2
#         ar_copy -= 1

#     else:
#         M[gen][:] = np.multiply(M_temp, np.subtract([1]*len(Mfc), Mfc))/sigma[gen]
#         F[gen][:] = np.multiply(F_temp, np.subtract([1]*len(Ffc), Ffc))/sigma[gen]
    
    M_1[gen][:] = np.multiply(M_1_temp, np.subtract([1]*len(Mfc), Mfc))/sigma_1[gen]
    F_1[gen][:] = np.multiply(F_1_temp, np.subtract([1]*len(Ffc), Ffc))/sigma_1[gen]
    M_2[gen][:] = np.multiply(M_2_temp, np.subtract([1]*len(Mfc), Mfc))/sigma_2[gen]
    F_2[gen][:] = np.multiply(F_2_temp, np.subtract([1]*len(Ffc), Ffc))/sigma_2[gen]
    
    M_1_sub = np.multiply(M_1[gen][:], mig_rate)
    F_1_sub = np.multiply(F_1[gen][:], mig_rate)
    
    M_2_sub = np.multiply(M_2[gen][:], mig_rate/pop_2_rel_size)
    F_2_sub = np.multiply(F_2[gen][:], mig_rate/pop_2_rel_size)
    
    M_1[gen][:] = M_1[gen][:] + np.multiply(M_2_sub, pop_2_rel_size) - M_1_sub
    F_1[gen][:] = F_1[gen][:] + np.multiply(F_2_sub, pop_2_rel_size) - F_1_sub
    
    M_2[gen][:] = M_2[gen][:] + np.divide(M_1_sub, pop_2_rel_size) - M_2_sub
    F_2[gen][:] = F_2[gen][:] + np.divide(F_1_sub, pop_2_rel_size) - F_2_sub
    
    if gen%ar_gen_freq == False and ar_copy > 0:
        ar_copy -= 1
        if len(ar_g) == 1:
        
            M_1[gen] = np.multiply(M_1[gen][:], 1-ar_f)
            F_1[gen] = np.multiply(F_1[gen][:], 1-ar_f)
            M_1[gen, ar_g] += ar_f/2
            F_1[gen,ar_g] += ar_f/2
        
        else:
            M_1[gen] = np.multiply(M_1[gen][:], 1-ar_f)
            F_1[gen] = np.multiply(F_1[gen][:], 1-ar_f)
            M_1[gen, ar_g[0]] += ar_f/2
            F_1[gen,ar_g[1]] += ar_f/2
        


# In[ ]:


## Plot Curves ##
plot_gens = 100

M = M_1
F = F_1

M_a_f_1 = np.zeros([numGens+1,len(M_a_f_s)])
F_a_f_1 = np.zeros([numGens+1,len(F_a_f_s)])
all_a_f_1 = np.zeros([numGens+1,len(all_a_f_s)])
M_p_f_1 = np.zeros([numGens+1,len(M_p_f_s)])
F_p_f_1 = np.zeros([numGens+1,len(F_p_f_s)])
all_p_f_1 = np.zeros([numGens+1,len(all_p_f_s)])
M_wt_f_1 = np.zeros(numGens+1)
F_wt_f_1 = np.zeros(numGens+1)

for f_i, frequency in enumerate(M_a_f_s):
    for g in range(numGens+1):
        M_a_f_1[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(M_p_f_s):
    for g in range(numGens+1):
        M_p_f_1[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(F_a_f_s):
    for g in range(numGens+1):
        F_a_f_1[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(F_p_f_s):
    for g in range(numGens+1):
        F_p_f_1[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(all_a_f_s):
    for g in range(numGens+1):
        all_a_f_1[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(all_p_f_s):
    for g in range(numGens+1):
        all_p_f_1[g][f_i] = eval(frequency)

# for pop 2
M = M_2
F = F_2

M_a_f_2 = np.zeros([numGens+1,len(M_a_f_s)])
F_a_f_2 = np.zeros([numGens+1,len(F_a_f_s)])
all_a_f_2 = np.zeros([numGens+1,len(all_a_f_s)])
M_p_f_2 = np.zeros([numGens+1,len(M_p_f_s)])
F_p_f_2 = np.zeros([numGens+1,len(F_p_f_s)])
all_p_f_2 = np.zeros([numGens+1,len(all_p_f_s)])
M_wt_f_2 = np.zeros(numGens+1)
F_wt_f_2 = np.zeros(numGens+1)

for f_i, frequency in enumerate(M_a_f_s):
    for g in range(numGens+1):
        M_a_f_2[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(M_p_f_s):
    for g in range(numGens+1):
        M_p_f_2[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(F_a_f_s):
    for g in range(numGens+1):
        F_a_f_2[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(F_p_f_s):
    for g in range(numGens+1):
        F_p_f_2[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(all_a_f_s):
    for g in range(numGens+1):
        all_a_f_2[g][f_i] = eval(frequency)
for f_i, frequency in enumerate(all_p_f_s):
    for g in range(numGens+1):
        all_p_f_2[g][f_i] = eval(frequency)
    


if XY == 1:
    # Phenotype Frequencies
    plt.subplot(221)
    plt.plot(all_p_f_1)
    #     plt.yscale('linear')
    plt.legend(all_a)
    plt.axis([0, plot_gens, 0, 1])
    plt.title('Phenotype Frequency')
    plt.grid(True)

    # Allele Frequencies
    plt.subplot(222)
    plt.plot(all_a_f_1)
    #     plt.yscale('linear')
    # plt.legend(all_a)
    plt.axis([0, plot_gens, 0, 1])
    plt.title('Allele Frequency')
    plt.grid(True)

    # Gendered Phenotype Frequencies
    plt.subplot(223)
    plt.plot(M_p_f_1)
    plt.plot(F_p_f_1)
    #     plt.yscale('linear')
    # plt.legend(M_a + F_a)
    plt.axis([0, plot_gens, 0, 1])
    # plt.title('Gendered Phenotype Frequency')
    plt.grid(True)

    # Gendered Allele Frequencies
    plt.subplot(224)
    plt.plot(M_a_f_1)
    plt.plot(F_a_f_1)
    #     plt.yscale('linear')
    plt.legend(M_a + F_a)
    plt.axis([0, plot_gens, 0, 1])
    # plt.title('Gendered Allele Frequency')
    plt.grid(True)
    
else:
    # Phenotype Frequencies
    plt.figure(figsize = (10,10))
    plt.plot(all_p_f_1, linewidth=7.0)
    #     plt.yscale('linear')
    plt.legend(all_a_1)
    plt.axis([0, plot_gens, 0, 1])
    plt.title('Phenotype Frequency')
    plt.grid(True)


# In[ ]:


plot_gens = 600
# Phenotype Frequencies
plt.figure(figsize = (10,10))
plt.subplot(211)
plt.plot(all_p_f_1[:,[0, 2]], linewidth=7.0)
plt.legend(['Cas9', 'Rescue'])
plt.axis([0, plot_gens, 0, 1])
plt.title('Phopulation 1, Linkage='+str(M_li_b[0])+', IF='+str(IF)+', FC='+str(alleles_m_fc[0])+', MR='+str(mig_rate)+', and '+str(ar)+' additional releases.')
plt.grid(True)

# Allele Frequencies
plt.subplot(212)
plt.plot(all_p_f_2[:,[0, 2]], linewidth=7.0)
plt.legend(['Cas9', 'Rescue'])
plt.axis([0, plot_gens, 0, 1])
plt.title('Population 2')
plt.grid(True)


# In[ ]:


print('Linkage='+str(M_li_b[0])+', IF='+str(IF)+', FC='+str(alleles_m_fc[0])+', MR='+str(mig_rate)+', and '+str(ar)+' additional releases')
print('The maximum genotype frequency for Cas9 in population 1 is: ' + str(max(all_p_f_1[0:200,0])))
print('The maximum genotype frequency for the Rescue in population 1 is: ' + str(max(all_p_f_1[0:200,2])))
print('The maximum genotype frequency for Cas9 in population 2 is: ' + str(max(all_p_f_2[0:200,0])))
print('The maximum genotype frequency for the Rescue in population 2 is: ' + str(max(all_p_f_2[0:200,2])))


# In[ ]:


# Phenotype Frequencies
plt.figure(figsize = (10,10))
plt.plot(all_p_f_1, linewidth=7.0)
#     plt.yscale('linear')
plt.legend(all_a)
plt.axis([0, 600, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


# Phenotype Frequencies
plt.figure(figsize = (10,10))
plt.plot(all_p_f_2, linewidth=7.0)
#     plt.yscale('linear')
plt.legend(all_a)
plt.axis([0, 600, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


for x in all_p_f_2[:, 2]:
    print(x)


# In[ ]:


all_a_f[:, 0]


# In[ ]:


for x in all_p_f[0:30, 2]:
    print(x)


# In[ ]:


all_a_f[400, :]


# In[ ]:


# Phenotype Frequencies
plt.figure(figsize = (10,10))
plt.plot(all_p_f, linewidth=7.0)
#     plt.yscale('linear')
plt.legend(all_a)
plt.axis([0, 20, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


# data_cas9bearing = np.array([66.66666667/2, 71.22807018, 68.68686869, 58.93416928, 49.77578475, 43.98148148, 41.33333333, 34.53815261, 34.33476395, 25.65217391, 30, 32.17054264, 31.87134503, 19.74248927, 27.08933718, 23.52941176, 28.43822844, 23.41920375, 23.46491228, 27.02702703, 26.05633803, 25.33333333, 28.20512821, 36.82170543, 33.72093023])
# data_cargobearing = np.array([66.66666667/2, 71.22807018, 83.83838384, 88.40125392, 91.47982063, 92.59259259, 94.22222222, 94.77911647, 98.2832618, 98.69565217, 98.69565217, 98.4496124, 99.41520468, 100, 100, 99.71988796, 99.76689977, 100, 100, 100, 100, 100, 100, 100, 100])
# gen = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]


data_cas9bearing = np.array([71.22807018, 68.68686869, 58.93416928, 49.77578475, 43.98148148, 41.33333333, 34.53815261, 34.33476395, 25.65217391, 30, 32.17054264, 31.87134503, 19.74248927, 27.08933718, 23.52941176, 28.43822844, 23.41920375, 23.46491228, 27.02702703, 26.05633803, 25.33333333, 28.20512821, 36.82170543, 33.72093023])
data_cargobearing = np.array([71.22807018, 83.83838384, 88.40125392, 91.47982063, 92.59259259, 94.22222222, 94.77911647, 98.2832618, 98.69565217, 98.69565217, 98.4496124, 99.41520468, 100, 100, 99.71988796, 99.76689977, 100, 100, 100, 100, 100, 100, 100, 100])
gen = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.]


# In[ ]:


# Phenotype Frequencies
plt.plot(gen,all_p_f[0:24], gen, data_cas9bearing/100, gen, data_cargobearing/100)
#     plt.yscale('linear')
plt.legend(all_a)
plt.axis([0, 10, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


all_a


# In[ ]:


# Phenotype Frequencies
plt.plot(gen, all_a_f[0:25], gen, data_cas9bearing/100)
#     plt.yscale('linear')
plt.legend(all_a)
plt.axis([0, 24, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)
plt.figure(figsize=(20,100))


# In[ ]:


second_introduction_males = M[14][:]
second_introduction_females = F[14][:]


# In[ ]:


all_a.append('Cas9-bearing data')
all_a.append('Cargo-bearing data')
all_a


# In[ ]:


all_p_f[0:30, 0]


# In[ ]:


M[0]


# In[ ]:


M[14]


# In[ ]:


for x in all_p_f[0:30, 2]:
    print(x)


# In[ ]:


# Phenotype Frequencies
plt.figure(figsize = (7,3))
plt.plot(all_p_f[:, 0:2], linewidth=7.0)
#     plt.yscale('linear')
plt.legend(['ClvR1', 'Clvr2'])
plt.axis([32, 43, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


# Phenotype Frequencies
plt.figure(figsize = (7,3))
plt.plot(all_p_f[:, 0:2], linewidth=7.0)
#     plt.yscale('linear')
plt.legend(['ClvR1', 'Clvr2'])
plt.axis([0, 10, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


# Phenotype Frequencies
plt.figure(figsize = (7,3))
plt.plot(all_p_f[:, 0:2], linewidth=7.0)
#     plt.yscale('linear')
plt.legend(['ClvR1', 'Clvr2'])
plt.axis([32, 43, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


# Allele Frequencies
plt.figure(figsize = (10,10))
plt.plot(all_a_f[:, 0:2], linewidth=7.0)
#     plt.yscale('linear')
plt.legend(['ClvR1', 'Clvr2'])
plt.axis([32, 52, 0, 1])
plt.title('Phenotype Frequency')
plt.grid(True)


# In[ ]:


all_a_f[32:52, 0:3]


# In[ ]:


all_a_f[32:52, 0:3]


# In[ ]:


all_p_f[32:52, 0:3]


# In[ ]:


# 2L
all_a_f[:, 0]


# In[ ]:


all_p_f[:, 1]


# In[ ]:


all_p_f[:, 0]


# In[ ]:





# In[ ]:


M_g_l[21]


# In[ ]:


F_g_l[42]


# In[ ]:


D_A_M[3]


# In[ ]:


M_g_l


# In[ ]:


end_time = time.clock() 
print(end_time-start_time)


# In[ ]:


def stochastic_sim(alleles, mods, sex_det, 
                   num_gens, intro, d_a, r_d, f_c, 
                   K, n_o, g_f):
    
    
    
    num_loci = len(alleles)
    
    diploid_loci = [list(product(allele, allele)) for allele in alleles]
    genotypes = [list(product(*diploid_loci)), [], []]
    
    geno_alleles = [[], [], []]
    for genotype in genotypes[0]:
        geno_alleles[0].append(allele_list(genotype, 'genotype'))
        
    adults = [[[]], [[]]]
    
    for genotype in intro:
        if genotype[1] == 'f':
            for individual in range(genotype[2]):
                adults[0][0].append(genotype[0])
        
        elif genotype[1] == 'm':
            for individual in range(genotype[2]):
                adults[1][0].append(genotype[0])
            
    for gen in range(1, num_gens+1):
        adults[0].append([])
        adults[1].append([])
        
    for 

