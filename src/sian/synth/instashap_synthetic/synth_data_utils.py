import numpy as np
import json
import math

from utils import powerset
import torch




def loadDataset(path):
    trnX = np.load(path+'trnX.npy')
    trnY = np.load(path+'trnY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnX,trnY,tstX,tstY



# TODO TODO TODO -- organize this with everything else




def get_faith_shap_coefficient(ell,s,t):
    if s==t:
        return 1.0

    coeff = float(math.comb(t-s-1,ell-s)) / float(math.comb(t+ell-1,ell)) * float(math.comb(ell+s-1,ell-1))

    if (ell-s)%2==0:
        return coeff
    else:
        return -coeff


def get_faith_shap_model_summation_coefficients(D,k,s):
    assert (D==10), 'only providing D=10,ell=2 hardcoded values for now'

    d10k1_coefficients = [19/55,16/495,13/1980,1/462,1/990,2/3465,1/4620,-1/990,-1/99,-8/55]
    d10k2_coefficients = [3/55,2/165,1/220,1/385,1/462,1/385,1/220,2/165,3/55]

    if k==1:
        pass
        return d10k1_coefficients[s]
    elif k==2:
        pass
        return d10k2_coefficients[s]
    else:
        raise Exception("we need k <= ll")

    return None



def calculate_exact_fnl_ANOVA(D,beta_dict,X_input_tensor, power_sum, rho,  k_max=1,precision_structure='linear_disconnected_pairs'):
    fnl_ANOVA_dict = {}
    shap_ANOVA_dict = {}
    faithshap_ANOVA_dict = {} #just manually do the 2D for now


    if k_max==1 and precision_structure=='linear_disconnected_pairs':
        for i in range(0,D,2):
            b1 = beta_dict[(i,)]
            b2 = beta_dict[(i+1,)]
            #print('b1',b1,'b2',b2)
            f1 = (b1+rho*b2) * X_input_tensor[:,i] / np.sqrt(power_sum)
            f2 = (rho*b1+b2) * X_input_tensor[:,i+1] / np.sqrt(power_sum)
            f12 = ((-b1*rho)*X_input_tensor[:,i+1] + (-b2*rho)*X_input_tensor[:,i]) / np.sqrt(power_sum)

            fnl_ANOVA_dict[(i,)]     = f1
            fnl_ANOVA_dict[(i+1,)]   = f2
            fnl_ANOVA_dict[(i,i+1)]  = f12

            phi1 = f1 + f12/2
            phi2 = f2 + f12/2
            shap_ANOVA_dict[i]   = phi1
            shap_ANOVA_dict[i+1] = phi2

            faithshap_ANOVA_dict[(i,)]     = f1
            faithshap_ANOVA_dict[(i+1,)]   = f2
            faithshap_ANOVA_dict[(i,i+1)]  = f12
            
        for i in range(D):
            for j in range(i+1,D):
                if (i,j) not in faithshap_ANOVA_dict:
                    faithshap_ANOVA_dict[(i,j)] = 0*faithshap_ANOVA_dict[(0,1)]

    elif k_max==2 and precision_structure=='linear_disconnected_pairs':
        for i in range(0,D,2):
            b1 = beta_dict[(i,)]
            b2 = beta_dict[(i+1,)]
            #print('b1',b1,'b2',b2)
            f1 = (b1+rho*b2) * X_input_tensor[:,i] / np.sqrt(power_sum)
            f2 = (rho*b1+b2) * X_input_tensor[:,i+1] / np.sqrt(power_sum)
            f12 = ((-b1*rho)*X_input_tensor[:,i+1] + (-b2*rho)*X_input_tensor[:,i]) / np.sqrt(power_sum)

            fnl_ANOVA_dict[(i,)]     = f1
            fnl_ANOVA_dict[(i+1,)]   = f2
            fnl_ANOVA_dict[(i,i+1)]  = f12


        for i1 in range(0,D,2):
            j1 = i1+1
            b12 = beta_dict[(i1,j1)]

            f1 = b12 * rho * (X_input_tensor[:,i1]**2 - 1)
            f2 = b12 * rho * (X_input_tensor[:,j1]**2 - 1)
            f12= b12 * (X_input_tensor[:,i1]*X_input_tensor[:,j1] - X_input_tensor[:,i1]**2 - X_input_tensor[:,j1]**2 + 1)

            for (tup,f) in [((i1,),f1),((j1,),f2),((i1,j1),f12)]:
                if tup in fnl_ANOVA_dict:
                    fnl_ANOVA_dict[tup] += f / np.sqrt(power_sum)
                else:
                    fnl_ANOVA_dict[tup] = f / np.sqrt(power_sum)


        for i1 in range(0,D):
            if i1%2==0:
                i2=i1+1
            else:
                i2=i1-1
            for k1 in range(int(i1/2)*2+2,D):
                if k1%2==0:
                    k2=k1+1
                else:
                    k2=k1-1

                b13 = beta_dict[(i1,k1)]

                f13 = b13 * X_input_tensor[:,i1] * X_input_tensor[:,k1]
                f14 = b13 * rho * X_input_tensor[:,i1] * X_input_tensor[:,k2]
                f23 = b13 * rho * X_input_tensor[:,i2] * X_input_tensor[:,k1]
                f24 = b13 * rho * rho * X_input_tensor[:,i2] * X_input_tensor[:,k2]

                f123 = -b13 * rho * X_input_tensor[:,i2] * X_input_tensor[:,k1]
                f124 = -b13 * rho * rho * X_input_tensor[:,i2] * X_input_tensor[:,k2]
                f134 = -b13 * rho * X_input_tensor[:,i1] * X_input_tensor[:,k2]
                f234 = -b13 * rho * rho * X_input_tensor[:,i2] * X_input_tensor[:,k2]

                f1234 = b13 * rho * rho * X_input_tensor[:,i2] * X_input_tensor[:,k2]

                for (tup,f) in [((i1,k1),f13),((i1,k2),f14),((i2,k1),f23),((i2,k2),f24),
                                ((i1,i2,k1),f123),((i1,i2,k2),f124),((i1,k1,k2),f134),((i2,k1,k2),f234),
                                ((i1,i2,k1,k2),f1234),]:
                    tup = tuple(sorted(list(tup)))
                    if tup in fnl_ANOVA_dict:
                        fnl_ANOVA_dict[tup] += f / np.sqrt(power_sum)
                    else:
                        fnl_ANOVA_dict[tup] = f / np.sqrt(power_sum)


        for i in range(0,D):      
            shap_ANOVA_dict[i] = 0.0
            
        for tup in fnl_ANOVA_dict:
            f_tup = fnl_ANOVA_dict[tup]
            for i in tup:
                shap_ANOVA_dict[i] += f_tup/len(tup)

        for i in range(0,D):      
            faithshap_ANOVA_dict[(i,)] = 0.0
            for j in range(i+1,D):      
                faithshap_ANOVA_dict[(i,j)] = 0.0

        for tup in fnl_ANOVA_dict:
            f_tup = fnl_ANOVA_dict[tup]
            for i in tup:
                faithshap_ANOVA_dict[(i,)] += f_tup * get_faith_shap_coefficient(2,1,len(tup))
            for i in tup:
                for j in tup:
                    if i<j:
                        faithshap_ANOVA_dict[(i,j)] += f_tup * get_faith_shap_coefficient(2,2,len(tup))


    else:
        raise Exception("Not implemented yet!!")

    return fnl_ANOVA_dict,shap_ANOVA_dict,faithshap_ANOVA_dict


def calculate_exact_power_sum(D,beta_dict, rho,  k_max=1,precision_structure='linear_disconnected_pairs'):

    if k_max==1 and precision_structure=='linear_disconnected_pairs':
        power_sum = 0
        for i in range(0,D,2):
            b1 = beta_dict[(i,)]
            b2 = beta_dict[(i+1,)]
            print('b1',b1,'b2',b2)
            power_sum+=b1*b1+2*rho*b1*b2+b2*b2

            print(power_sum)


    elif k_max==2 and precision_structure=='linear_disconnected_pairs':
        power_sum = 0
        for i in range(0,D,2):
            b1 = beta_dict[(i,)]
            b2 = beta_dict[(i+1,)]
            #print('b1',b1,'b2',b2)
            power_sum+=b1*b1+2*rho*b1*b2+b2*b2
            #print(power_sum)

        for i1 in range(0,D,2):   #USING THE RIGHT CORRECTION FOR (x1*x2-rho)
            for i2 in range(0,D,2):
                b12 = beta_dict[(i1,i1+1)]
                b34 = beta_dict[(i2,i2+1)]
                if i1==i2:
                    power_sum+=b12*b34*(1+rho*rho)

        def are_rho_sim_under_disconn(i1,i2):
            if i1==i2+1:
                if i2%2==0:
                    return True
            if i2==i1+1:
                if i1%2==0:
                    return True
            return False

        for i1 in range(0,D):
            for k1 in range(int(i1/2)*2+2,D):
                for i2 in range(0,D):
                    for k2 in range(int(i2/2)*2+2,D):
                        b12 = beta_dict[(i1,k1)]
                        b34 = beta_dict[(i2,k2)]
                        if i1==i2:
                            if k1==k2:
                                power_sum+=b12*b34*(1)
                            elif are_rho_sim_under_disconn(k1,k2):
                                power_sum+=b12*b34*(rho)
                        elif are_rho_sim_under_disconn(i1,i2):
                            if k1==k2:
                                power_sum+=b12*b34*(rho)
                            elif are_rho_sim_under_disconn(k1,k2):
                                power_sum+=b12*b34*(rho*rho)
                        #print(power_sum)
        print('power_sum')
        print(power_sum) #still print once at the end

    else:
        raise Exception("power_sum  not implemented yet (for this k)")

    return power_sum



def calculate_model_exact_fnl_ANOVA(D,target_net,X_input_tensor):
    cum_fnl_ANOVA_dict = {}
    shap_ANOVA_dict = {}
    for i in range(D): #need to change later for faith shap
        shap_ANOVA_dict[i] = 0.0

    powerset_tuples=list(powerset(range(D)))  
    target_net.precompute_on(X_input_tensor)


    with torch.no_grad(): #maybe also need this for OOM (bc I think it was holding all the gradients)
        for tup in powerset_tuples:
            for i in range(D):
                if i not in tup:

                    s_batch = torch.zeros_like( X_input_tensor )
                    for ii in tup:
                        s_batch[:,ii] = 1.0
                    logits = target_net((X_input_tensor,s_batch))
                    g_tup = logits[0]+logits[1]
                    gain = -g_tup.detach().cpu()
                    del g_tup

                    s_batch[:,i] = 1.0
                    logits = target_net((X_input_tensor,s_batch))
                    g_tup_plus_i = logits[0]+logits[1]
                    gain += g_tup_plus_i.detach().cpu()
                    del g_tup_plus_i #maybe this is enough

                    shap_ANOVA_dict[i] += gain * 1.0/float(math.comb(D-1,len(tup)))/float(D)

                    pass


    for i in range(D):
        shap_ANOVA_dict[i] = shap_ANOVA_dict[i][:,0].to(X_input_tensor.device)


    target_net.precompute_off()
    return None,shap_ANOVA_dict


def calculate_model_exact_faithSHAP_2D_ANOVA(D,target_net,X_input_tensor):
    cum_fnl_ANOVA_dict = {}
    shap_ANOVA_dict = {}
    faithshap_ANOVA_dict = {}

    
    for i in range(D): 
        faithshap_ANOVA_dict[(i,)] = 0.0
        for j in range(i+1,D):
            faithshap_ANOVA_dict[(i,j)] = 0.0

    powerset_tuples=list(powerset(range(D)))  
    target_net.precompute_on(X_input_tensor)

    with torch.no_grad(): #maybe also need this for OOM (bc I think it was holding all the gradients)
        for tup in powerset_tuples:
            ###print(tup)
            for i in range(D):
                if i not in tup:
                    s_batch = torch.zeros_like( X_input_tensor )
                    for ii in tup:
                        s_batch[:,ii] = 1.0
                    logits = target_net((X_input_tensor,s_batch))
                    g_tup = logits[0]+logits[1]
                    gain = -g_tup.detach().cpu()
                    del g_tup

                    s_batch[:,i] = 1.0
                    logits = target_net((X_input_tensor,s_batch))
                    g_tup_plus_i = logits[0]+logits[1]
                    gain += g_tup_plus_i.detach().cpu()
                    del g_tup_plus_i 

                    faithshap_ANOVA_dict[(i,)] += gain * get_faith_shap_model_summation_coefficients(D,1,len(tup))

                    for j in range(i+1,D):
                        if j not in tup:

                            s_batch = torch.zeros_like( X_input_tensor )
                            for ii in tup:
                                s_batch[:,ii] = 1.0
                            logits = target_net((X_input_tensor,s_batch))
                            g_tup = logits[0]+logits[1]
                            gain = g_tup.detach().cpu()
                            del g_tup

                            s_batch[:,i] = 1.0
                            logits = target_net((X_input_tensor,s_batch))
                            g_tup_plus_i = logits[0]+logits[1]
                            gain -= g_tup_plus_i.detach().cpu()
                            del g_tup_plus_i 

                            s_batch[:,j] = 1.0
                            logits = target_net((X_input_tensor,s_batch))
                            g_tup_plus_ij = logits[0]+logits[1]
                            gain += g_tup_plus_ij.detach().cpu()
                            del g_tup_plus_ij 

                            s_batch[:,i] = 0.0
                            logits = target_net((X_input_tensor,s_batch))
                            g_tup_plus_j = logits[0]+logits[1]
                            #gain += g_tup_plus_j.detach().cpu()
                            gain -= g_tup_plus_j.detach().cpu()
                            del g_tup_plus_j 

                            faithshap_ANOVA_dict[(i,j)] += gain * get_faith_shap_model_summation_coefficients(D,2,len(tup))


                    pass


    for i in range(D): 
        faithshap_ANOVA_dict[(i,)] = faithshap_ANOVA_dict[(i,)][:,0].to(X_input_tensor.device) 
        for j in range(i+1,D):
            faithshap_ANOVA_dict[(i,j)] = faithshap_ANOVA_dict[(i,j)][:,0].to(X_input_tensor.device)

    target_net.precompute_off()
    return None,None,faithshap_ANOVA_dict






def load_and_split_dataset(dataset_params, trnval_split_seed=0):
    pass

    thing = (dataset_params['n'],dataset_params['d'],dataset_params['kMax'],dataset_params['rho'],)
    synth_data_path_prefix = "n{:d}_d{:d}_kMax{:d}_rho_{:.4f}_".format(*thing)
    thing = (dataset_params['noiseRat'],dataset_params['datagen_seed'],)
    synth_data_path_prefix+="noiseRat_{:.2f}_seed_{:d}__".format(*thing)
    thing = (dataset_params['precision_structure'],dataset_params['beta_type'],)
    synth_data_path_prefix+="{:s}__{:s}__/".format(*thing)

    data_path = '../data/synth_data/'+synth_data_path_prefix
    data_hyper_dict = {}
    with open(data_path+'hyperparameter.json', 'r', encoding='utf-8') as f:
        data_hyper_dict = json.load(f)
    beta_dict_stringified=data_hyper_dict['beta_dict']
    beta_dict = {}
    for thing in beta_dict_stringified:
        tup=tuple(thing['key'])
        beta=(thing['val'])
        beta_dict[tup] = beta
    D=10


    trnX1,trnY1,    tstX,tstY = loadDataset(data_path)
    trnX1_og = trnX1    
    trnY1_og = trnY1


    #TODO: this part is difficult (mathemtiaclly)
    power_sum = calculate_exact_power_sum(D,beta_dict, dataset_params['rho'],
                                            k_max=dataset_params['kMax'],precision_structure=dataset_params['precision_structure'])
    #doing the exact form to be easier to work with (cant yet generalize to GAM2 etc)
    trnY1_og[:,0:1] = trnY1_og[:,1:2]/np.sqrt(power_sum)

    print()
    print('and btw')
    print(np.var(trnY1_og[:,1:2]))
    print(np.mean(trnY1_og[:,1:2]**2))
    print(np.mean(trnY1_og[:,1:2]))


    np.random.seed(trnval_split_seed)
    mn=0
    M_NUM = trnX1_og.shape[0]
    rand_indices = np.random.permutation(M_NUM)
    per = .7
    M_TRN_NUM = int(M_NUM*per)
    trnX1 = trnX1_og[rand_indices[:M_TRN_NUM]]
    trnY1 = trnY1_og[rand_indices[:M_TRN_NUM]]
    valX1 = trnX1_og[rand_indices[M_TRN_NUM:]]
    valY1 = trnY1_og[rand_indices[M_TRN_NUM:]]

    data_hyper_dict['kMax']=dataset_params['kMax']
    presplit_training_data = {
        'data_hyper_dict' : data_hyper_dict,
        'beta_dict' : beta_dict,
        'trnval_split_seed' : trnval_split_seed,

        'trnX1' : trnX1,
        'trnY1' : trnY1,
        'valX1' : valX1,
        'valY1' : valY1,

    }
    return presplit_training_data























