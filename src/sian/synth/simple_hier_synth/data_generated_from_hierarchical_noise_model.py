


import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from itertools import chain, combinations
from itertools import product

# TODO: pull this out somewhere
TENSOR_COMPUTE_STYLE = "N_batching_dataset"
# TENSOR_COMPUTE_STYLE = "ituple_batching_indices"




#TODO  -- need to make sure everything has a meta seed and a generation seed


#TODO -- idk how to avoid circular imports :(
# from .purification_helpers import purify_one_step_discrete
# from .purification_helpers import purify_discrete
# from .purification_helpers import shuffled_subset
# from .purification_helpers import strongest_heredity_subset



def gaussian_copulize(X):
    return erf(X)    



def get_gaussian_hierarchical_data(N, rho=0.7071,   S=3, B=2, depth=2):
    ZZ = np.random.randn(N,S)
    curr = ZZ
    history = []
    for d in range(depth):
        history.append(curr)
        _,f = curr.shape
    
        noise = np.random.randn(*curr.shape,B)
        next = np.sqrt(rho) * curr[...,None]  +  np.sqrt(1-rho) * noise
        curr = next.reshape(-1,f*B)
    
    history.append(curr)
    stacked_history = np.concatenate(history,axis=-1)
    return stacked_history


def get_uniform_hierarchical_data(N, rho=0.7071,   S=3, B=2, depth=2):
    stacked_history = get_gaussian_hierarchical_data(N, rho,S=S,B=B,depth=depth)
    new_stacked_history = gaussian_copulize(stacked_history)
    return new_stacked_history


#NOTE: why is this one so much slower than the others
def get_discrete_hierarchical_data(N, I_k, eps=0.10,   S=3, B=2, depth=2):
    ZZ = (np.random.rand(N,S)*I_k).astype(int)
    curr = ZZ
    history = []
    for d in range(depth):
        history.append(curr)
        _,f = curr.shape
    
        flip = (np.random.rand(*curr.shape,B) <= eps)
        noise = (np.random.rand(*curr.shape,B)*I_k).astype(int)
        next = (1-flip) * curr[...,None] + flip * noise
        curr = next.reshape(-1,f*B)
        
        if False: #this doesnt work, but it was an attempt to make it faster
            flip = (np.random.rand(B,*curr.shape) <= eps)
            noise = (np.random.rand(B,*curr.shape)*I_k).astype(int)
    # pass
    #         print('flip',flip.shape)
    #         print('noise',noise.shape)
    #         print('curr',curr.shape)
    #         print('    ',np.tile(curr, (B,1,1)).shape)
            ###next = (1-flip) * curr[...,None] + flip * noise
            next = (1-flip) * np.tile(curr, (B,1,1)) + flip * noise
            curr = next.transpose((2,0,1)).reshape(-1,f*B)
        
    history.append(curr)
    stacked_history = np.concatenate(history,axis=-1)
    return stacked_history
    









class Continuous_HierarchicalNoiseModel_InputDataGenerator():
    def __init__(self, feature_type, rho, sources=3, branching=2, depth=2):

        self.D = sources * branching**depth
        self.generator_meta_parameters = {} #idk which way it should be TBH
        self.generator_parameters = {
            'feature_type' : feature_type, # 'cts_gauss' or 'cts_unif'
            'rho' : rho,
            
            'sources' : sources,
            'branching' : branching,
            'depth' : depth,
        }



    # def generate_samples(self, N):
    def generate_samples(self, N, return_full_hiearchy=False):

        X_dict = {}

        rho, S, B, depth = self.generator_parameters['rho'], self.generator_parameters['sources'], self.generator_parameters['branching'],self.generator_parameters['depth']
        if self.generator_parameters['feature_type'] == 'cts_gauss':
            stacked_history = get_gaussian_hierarchical_data(N, rho=rho,S=S,B=B,depth=depth)
        elif self.generator_parameters['feature_type'] == 'cts_unif':
            stacked_history = get_uniform_hierarchical_data(N, rho=rho,S=S,B=B,depth=depth)
        else:
            raise Exception(f"feature_type={feature_type} not supported")
        
        if return_full_hiearchy:
            return stacked_history
         
        for d in range(self.D):
            d2 = stacked_history.shape[1]-self.D+d
            X_d = stacked_history[:,d2]
            # X_dict[d] = X_d
            X_dict[d] = X_d[:,None]  #09/20/25 @ 2:25pm
            # print(X_d.shape)

        return X_dict


class Discrete_HierarchicalNoiseModel_InputDataGenerator():
    def __init__(self, I_k, eps, sources=3, branching=2, depth=2):

        if depth==0:
            branching = 1 #avoid this accidentally mattering in below equation
        self.D = sources * branching**depth
        self.I_k = I_k
        self.generator_meta_parameters = {}
        self.generator_parameters = {
            'feature_type' : 'disc_unif',
            'eps' : eps,
            
            'sources' : sources,
            'branching' : branching,
            'depth' : depth,
        }


    def generate_samples(self, N, return_full_hiearchy=False):

        X_dict = {}

        eps, S, B, depth = self.generator_parameters['eps'], self.generator_parameters['sources'], self.generator_parameters['branching'],self.generator_parameters['depth']
        if self.generator_parameters['feature_type'] == 'disc_unif':
            stacked_history = get_discrete_hierarchical_data(N,self.I_k, eps=eps,S=S,B=B,depth=depth)
        else:
            raise Exception(f"feature_type={feature_type} not supported")

        if return_full_hiearchy:
            return stacked_history

        for d in range(self.D):
            d2 = stacked_history.shape[1]-self.D+d
            X_d_int = stacked_history[:,d2]
            X_d = np.zeros((N,self.I_k),dtype=int)
            X_d[np.arange(N),X_d_int] = 1
            X_dict[d] = X_d
            # print(X_d.shape)

        return X_dict















class DiscreteToContinuous_PurifiedCoefficients3D_OutputDataFunction():
    def __init__(self, D, I_k, max_tuples_per_band, band_strengths, anova_purification_type, dgp_x, heredity_style='nonhereditary'):

        self.D = D
        self.I_k = I_k
        self.generator_meta_parameters = {
            'max_tuples_per_band' : max_tuples_per_band,
            'band_strengths' : band_strengths,
            'anova_purification_type' : anova_purification_type,
        }
        self.generator_parameters = {}
        
        # purification_type = "purification_off"
        # purification_type = "marginal_ANOVA"
        # purification_type = "conditional_Sobol_ANOVA"
        # purification_type = "conditional_Hooker_ANOVA"
        

    
        for k in max_tuples_per_band:
            if max_tuples_per_band[k]==0:
                assert band_strengths[k]==0, "cannot have a positive band with no interaction terms"
        
        singles = [(i,) for i in range(D)]
        pairs = list(combinations((range(D)),2))
        triples = list(combinations((range(D)),3))

        if heredity_style=="nonhereditary":
            singles = shuffled_subset(singles, max_tuples_per_band[1])
            pairs   = shuffled_subset(pairs,   max_tuples_per_band[2])
            triples = shuffled_subset(triples, max_tuples_per_band[3])
        elif heredity_style=="strongest_heredity":
            singles = shuffled_subset(singles, max_tuples_per_band[1])
            print('singles',singles)
            pairs   = strongest_heredity_subset(pairs,   singles, max_tuples_per_band[2])
            triples = strongest_heredity_subset(triples, pairs,   max_tuples_per_band[3])
        elif heredity_style=="unshuffled":
            singles = singles[:max_tuples_per_band[1]]
            pairs   =   pairs[:max_tuples_per_band[2]]
            triples = triples[:max_tuples_per_band[3]]
        else:
            raise NotImplementedError(f'heredity_style={heredity_style}')
            




        all_indices = singles + pairs + triples
        self.generator_parameters['all_indices'] = all_indices


        #TODO: AHHHHHHHHH How to deal with these necessity
        # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.0, sources=D, branching=0, depth=0)
        # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.50, sources=1, branching=D, depth=1)
        N = 1000*1000 
        X_dict = dgp_x.generate_samples(N)
        X_onehot = np.concatenate([X_dict[d] for d in range(D)],axis=1)
        print('X_onehot',X_onehot.shape)
        

        beta_dict = {}
        beta_power_bands = {}
        for K in range(3):
            beta_power_bands[K+1] = 0.0
        for tup in all_indices:
            print('\t',tup)
            beta_tt = np.random.randn(*([I_k]*len(tup)))
            ###beta_tt = np.random.randint(10,size=([I_k]*len(tup)))
            beta_dict[tup] = beta_tt

            ASSUME_UNIFORM_DISTRIBUTION = False
            if ASSUME_UNIFORM_DISTRIBUTION: #only works for uniform distribution
                complete_semimasked_U = np.ones_like(beta_tt)
                complete_semimasked_U = complete_semimasked_U / np.sum(complete_semimasked_U)
            else: 
                if TENSOR_COMPUTE_STYLE=="N_batching_dataset": #supposedly terrible for memory
                    preU = 0.0
                    N_batch_size = 100*1000
                    
                    for nn in range(11):
                        preX = np.ones(())
                        for kk,k in enumerate(tup):
                            preX = preX[...,None] * X_onehot[nn*N_batch_size:(nn+1)*N_batch_size,I_k*k:I_k*(k+1)][ tuple([slice(None)]+[(None)]*(kk)) ]
                        preU += np.sum(preX, axis=0)
                    complete_semimasked_U = preU / X_onehot.shape[0]
                elif TENSOR_COMPUTE_STYLE=="N_batching_dataset": 
                    raise NotImplementedError("marginals not implemented for this yet")
                else:
                    raise ValueError(f"TENSOR_COMPUTE_STYLE={TENSOR_COMPUTE_STYLE} not supported")
            # print('complete_semimasked_U')
            # print(complete_semimasked_U)

            beta_tt_adjusted = purify_discrete(beta_tt, complete_semimasked_U, anova_purification_type)
            if ASSUME_UNIFORM_DISTRIBUTION:
                pass
                #beta_power_bands[len(tup)] += np.mean(beta_tt_adjusted**2) #wont work for correlated -- maybe remove
            else:
                beta_power_bands[len(tup)] += np.sum(complete_semimasked_U*beta_tt_adjusted**2)
            beta_dict[tup] = beta_tt_adjusted

        # print('beta_power_bands')
        # for K in range(3):
        #     print('\t',K+1,beta_power_bands[K+1])
        # print()
                
        final_beta_dict = {} #rescaled version
        for tup in all_indices[0:]:
            final_beta_tt = beta_dict[tup] * np.sqrt( band_strengths[len(tup)] / beta_power_bands[len(tup)] )
            final_beta_dict[tup] = final_beta_tt

        self.generator_parameters['final_beta_dict'] = final_beta_dict
        self.generator_parameters['beta_dict'] = beta_dict



    
    # # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.0, sources=D, branching=0, depth=0)
    # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.50, sources=1, branching=D, depth=1)
    # X_dict = dgp_x.generate_samples(N)
    # X_onehot = np.concatenate([X_dict[d] for d in range(D)],axis=1)
    # print('X_onehot',X_onehot.shape)

    # def generate_output_samples(self, X):
    def generate_output_samples(self, X_onehot):
        pass
        N = X_onehot[0].shape[0]

        final_beta_dict = self.generator_parameters['final_beta_dict']
        all_indices = self.generator_parameters['all_indices']
        I_k = self.I_k
        
        full_y = 0.0

        tup_size = 1
        for tup in all_indices:
            if len(tup) != tup_size:
                tup_size = len(tup)
                print()

            #final_beta_tt = beta_dict[tup] / np.sqrt(beta_power_bands[len(tup)]) / np.sqrt(3.)
            final_beta_tt = final_beta_dict[tup] 

            
            if TENSOR_COMPUTE_STYLE == "ituple_batching_indices":
                preY = np.zeros(X_onehot.shape[0])
                for i_tup in product(*([list(range(I_k))]*len(tup))):
                    print('\t',i_tup)
        
        
                    inds = np.ones(X_onehot.shape[0],dtype=bool)
                    for kk,k in enumerate(tup):
                        #NOTE: this indexing is not general and depends on the [I_k]^d form.
                        inds = np.logical_and(inds, X_onehot[:,I_k*k+i_tup[kk]]==1)
                    preY[inds] += final_beta_tt[i_tup]

            elif TENSOR_COMPUTE_STYLE == "N_batching_dataset":
                #NOTE: NEED TO ADAPTIVELY SELECT "N_BS" to fit on the GPU properly (tuple by tuple in all likelihood)
                preY = np.zeros(X_onehot.shape[0])
                N_batch_size = 100*1000
                
                for nn in range(11):
                    preX = np.ones(())
                    for kk,k in enumerate(tup):
                        preX = preX[...,None] * X_onehot[nn*N_batch_size:(nn+1)*N_batch_size,I_k*k:I_k*(k+1)][ tuple([slice(None)]+[(None)]*(kk)) ]
                        # print('preX',preX.shape)
                    nearPreY = final_beta_tt[None] * preX
                    preY[nn*N_batch_size:(nn+1)*N_batch_size] = np.sum(nearPreY, axis=tuple(range(1,len(tup)+1)))

            else:
                raise ValueError(f"TENSOR_COMPUTE_STYLE={TENSOR_COMPUTE_STYLE} not supported")


            full_y +=  preY
            # print('full_y','var',np.var(full_y))
        print('full_y','var',np.var(full_y))
        return full_y








class DiscreteToContinuous_PurifiedCoefficients_kD_OutputDataFunction():
    def __init__(self, D, I_k, max_tuples_per_band, band_strengths, anova_purification_type, dgp_x, heredity_style='nonhereditary'):

        self.D = D
        self.I_k = I_k
        self.K = len(max_tuples_per_band)
        self.generator_meta_parameters = {
            'max_tuples_per_band' : max_tuples_per_band,
            'band_strengths' : band_strengths,
            'anova_purification_type' : anova_purification_type,
        }
        self.generator_parameters = {}
        
        # purification_type = "purification_off"
        # purification_type = "marginal_ANOVA"
        # purification_type = "conditional_Sobol_ANOVA"
        # purification_type = "conditional_Hooker_ANOVA"
        

    
        for k in max_tuples_per_band:
            if max_tuples_per_band[k]==0:
                assert band_strengths[k]==0, "cannot have a positive band with no interaction terms"
        

        all_indices = []
        if heredity_style=="nonhereditary":
            for k in range(1,self.K+1):
                cands_k  = list(combinations((range(D)),k))
                tuples_k = shuffled_subset(cands_k, max_tuples_per_band[k])
                all_indices.extend( tuples_k )
        elif heredity_style=="strongest_heredity":
            cands_1  = list(combinations((range(D)),1))
            tuples_1 = shuffled_subset(cands_1, max_tuples_per_band[1])
            all_indices.extend( tuples_1 )

            prev_tuples = copy.deepcopy(tuples_1)
            for k in range(2,self.K+1):
                cands_k  = list(combinations((range(D)),k))
                tuples_k = strongest_heredity_subset(cands_k, prev_tuples, max_tuples_per_band[k])
                all_indices.extend( tuples_k )
                prev_tuples = copy.deepcopy(tuples_k)
        elif heredity_style=="unshuffled":
            for k in range(1,self.K+1):
                cands_k  = list(combinations((range(D)),k))
                tuples_k = cands_k[:max_tuples_per_band[k]]
                all_indices.extend( tuples_k )
        else:
            raise NotImplementedError(f'heredity_style={heredity_style}')
            




        self.generator_parameters['all_indices'] = all_indices
        print('all_indices',all_indices)


        #TODO: AHHHHHHHHH How to deal with these necessity
        # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.0, sources=D, branching=0, depth=0)
        # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.50, sources=1, branching=D, depth=1)
        N = 1000*1000 
        X_dict = dgp_x.generate_samples(N)
        X_onehot = np.concatenate([X_dict[d] for d in range(D)],axis=1)
        print('X_onehot',X_onehot.shape)
        

        beta_dict = {}
        beta_power_bands = {}
        for k in range(self.K):
            beta_power_bands[k+1] = 0.0
        for tup in all_indices:
            print('\t',tup)
            beta_tt = np.random.randn(*([I_k]*len(tup)))
            ### beta_tt = np.random.randint(10,size=([I_k]*len(tup))) #only back off at 09/20/25 @ 2:15pm
            beta_dict[tup] = beta_tt

            ASSUME_UNIFORM_DISTRIBUTION = False
            if ASSUME_UNIFORM_DISTRIBUTION: #only works for uniform distribution
                complete_semimasked_U = np.ones_like(beta_tt)
                complete_semimasked_U = complete_semimasked_U / np.sum(complete_semimasked_U)
            else: 
                if TENSOR_COMPUTE_STYLE=="N_batching_dataset": #supposedly terrible for memory
                    preU = 0.0
                    N_batch_size = 100*1000
                    
                    for nn in range(11):
                        preX = np.ones(())
                        for kk,k in enumerate(tup):
                            preX = preX[...,None] * X_onehot[nn*N_batch_size:(nn+1)*N_batch_size,I_k*k:I_k*(k+1)][ tuple([slice(None)]+[(None)]*(kk)) ]
                        preU += np.sum(preX, axis=0)
                    complete_semimasked_U = preU / X_onehot.shape[0]
                elif TENSOR_COMPUTE_STYLE=="N_batching_dataset": 
                    raise NotImplementedError("marginals not implemented for this yet")
                else:
                    raise ValueError(f"TENSOR_COMPUTE_STYLE={TENSOR_COMPUTE_STYLE} not supported")
            # print('complete_semimasked_U')
            # print(complete_semimasked_U)

            beta_tt_adjusted = purify_discrete(beta_tt, complete_semimasked_U, anova_purification_type)
            if ASSUME_UNIFORM_DISTRIBUTION:
                pass
                #beta_power_bands[len(tup)] += np.mean(beta_tt_adjusted**2) #wont work for correlated -- maybe remove
            else:
                beta_power_bands[len(tup)] += np.sum(complete_semimasked_U*beta_tt_adjusted**2)
            beta_dict[tup] = beta_tt_adjusted

        # print('beta_power_bands')
        # for k in range(self.K):
        #     print('\t',k+1,beta_power_bands[k+1])
        # print()
                
        final_beta_dict = {} #rescaled version
        for tup in all_indices[0:]:
            final_beta_tt = beta_dict[tup] * np.sqrt( band_strengths[len(tup)] / beta_power_bands[len(tup)] )
            final_beta_dict[tup] = final_beta_tt

        self.generator_parameters['final_beta_dict'] = final_beta_dict
        self.generator_parameters['beta_dict'] = beta_dict



    
    def generate_output_samples(self, X_onehot):
        pass
        N = X_onehot[0].shape[0]

        final_beta_dict = self.generator_parameters['final_beta_dict']
        all_indices = self.generator_parameters['all_indices']
        I_k = self.I_k
        
        full_y = 0.0

        tup_size = 1
        for tup in all_indices:
            if len(tup) != tup_size:
                tup_size = len(tup)
                print()

            #final_beta_tt = beta_dict[tup] / np.sqrt(beta_power_bands[len(tup)]) / np.sqrt(3.)
            final_beta_tt = final_beta_dict[tup] 
            # print('\t','final_beta_tt',np.mean(np.abs(final_beta_tt)))
            # print('\t','final_beta_tt',np.mean((final_beta_tt**2)))
            # xd=final_beta_tt.reshape(-1)
            # plt.scatter(np.arange(xd.shape[0]),xd); plt.plot(np.arange(xd.shape[0]), xd); plt.ylim(-0.5,0.5); plt.show();

            
            if TENSOR_COMPUTE_STYLE == "ituple_batching_indices":
                preY = np.zeros(X_onehot.shape[0])
                for i_tup in product(*([list(range(I_k))]*len(tup))):
                    print('\t',i_tup)
        
        
                    inds = np.ones(X_onehot.shape[0],dtype=bool)
                    for kk,k in enumerate(tup):
                        #NOTE: this indexing is not general and depends on the [I_k]^d form.
                        inds = np.logical_and(inds, X_onehot[:,I_k*k+i_tup[kk]]==1)
                    preY[inds] += final_beta_tt[i_tup]

            elif TENSOR_COMPUTE_STYLE == "N_batching_dataset":
                #NOTE: NEED TO ADAPTIVELY SELECT "N_BS" to fit on the GPU properly (tuple by tuple in all likelihood)
                preY = np.zeros(X_onehot.shape[0])
                N_batch_size = 100*1000
                
                for nn in range(11):
                    preX = np.ones(())
                    for kk,k in enumerate(tup):
                        preX = preX[...,None] * X_onehot[nn*N_batch_size:(nn+1)*N_batch_size,I_k*k:I_k*(k+1)][ tuple([slice(None)]+[(None)]*(kk)) ]
                        # print('preX',preX.shape)
                    nearPreY = final_beta_tt[None] * preX
                    preY[nn*N_batch_size:(nn+1)*N_batch_size] = np.sum(nearPreY, axis=tuple(range(1,len(tup)+1)))

            else:
                raise ValueError(f"TENSOR_COMPUTE_STYLE={TENSOR_COMPUTE_STYLE} not supported")


            full_y +=  preY
            print('partial_y','var',np.var(full_y))
        print('full_y','var',np.var(full_y))
        return full_y









class ContinuousToContinuous_LinearCoefficients_kD_OutputDataFunction():
    def __init__(self, D, max_tuples_per_band, band_strengths, anova_purification_type, dgp_x, heredity_style='nonhereditary'):

        anova_purification_type = 'unpurified_ANOVA'

        self.D = D
        #self.I_k = I_k
        self.K = len(max_tuples_per_band)
        self.generator_meta_parameters = {
            'max_tuples_per_band' : max_tuples_per_band,
            'band_strengths' : band_strengths,
            'anova_purification_type' : anova_purification_type,
        }
        self.generator_parameters = {}
        
        # purification_type = "purification_off"
        # purification_type = "marginal_ANOVA"
        # purification_type = "conditional_Sobol_ANOVA"
        # purification_type = "conditional_Hooker_ANOVA"
        

    
        for k in max_tuples_per_band:
            if max_tuples_per_band[k]==0:
                assert band_strengths[k]==0, "cannot have a positive band with no interaction terms"
        

        all_indices = []
        if heredity_style=="nonhereditary":
            for k in range(1,self.K+1):
                cands_k  = list(combinations((range(D)),k))
                tuples_k = shuffled_subset(cands_k, max_tuples_per_band[k])
                all_indices.extend( tuples_k )
        elif heredity_style=="strongest_heredity":
            cands_1  = list(combinations((range(D)),1))
            tuples_1 = shuffled_subset(cands_1, max_tuples_per_band[1])
            all_indices.extend( tuples_1 )

            prev_tuples = copy.deepcopy(tuples_1)
            for k in range(2,self.K+1):
                cands_k  = list(combinations((range(D)),k))
                tuples_k = strongest_heredity_subset(cands_k, prev_tuples, max_tuples_per_band[k])
                all_indices.extend( tuples_k )
                prev_tuples = copy.deepcopy(tuples_k)
        elif heredity_style=="unshuffled":
            for k in range(1,self.K+1):
                cands_k  = list(combinations((range(D)),k))
                tuples_k = cands_k[:max_tuples_per_band[k]]
                all_indices.extend( tuples_k )
        else:
            raise NotImplementedError(f'heredity_style={heredity_style}')
            




        self.generator_parameters['all_indices'] = all_indices
        print('all_indices',all_indices)


        #TODO: AHHHHHHHHH How to deal with these necessity
        # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.0, sources=D, branching=0, depth=0)
        # dgp_x = Discrete_HierarchicalNoiseModel_InputDataGenerator(I_k, eps=0.50, sources=1, branching=D, depth=1)
        N = 1000*1000 
        X_dict = dgp_x.generate_samples(N)
        X_onehot = np.concatenate([X_dict[d] for d in range(D)],axis=1)
        print('X_onehot',X_onehot.shape)
        

        beta_dict = {}
        beta_power_bands = {}
        for k in range(self.K):
            beta_power_bands[k+1] = 0.0
        for tup in all_indices:
            print('\t',tup)
            # beta_tt = np.random.randn(*([I_k]*len(tup)))
            #### beta_tt = np.random.randint(10,size=([I_k]*len(tup))) 
            beta_tt = np.random.randn( *([1]*len(tup)) )
            beta_dict[tup] = beta_tt

            ASSUME_UNIFORM_DISTRIBUTION = False
            if ASSUME_UNIFORM_DISTRIBUTION: #only works for uniform distribution
                complete_semimasked_U = np.ones_like(beta_tt)
                complete_semimasked_U = complete_semimasked_U / np.sum(complete_semimasked_U)
            else: 
                if TENSOR_COMPUTE_STYLE=="N_batching_dataset": #supposedly terrible for memory
                    preU = 0.0
                    N_batch_size = 100*1000

                    I_k = 1 #09/20/25
                    preY = np.zeros(X_onehot.shape[0])  #09/20/25
                    
                    for nn in range(11):
                        preX = np.ones(())
                        for kk,k in enumerate(tup):
                            preX = preX[...,None] * X_onehot[nn*N_batch_size:(nn+1)*N_batch_size,I_k*k:I_k*(k+1)][ tuple([slice(None)]+[(None)]*(kk)) ]
                        #preU += np.sum(preX, axis=0)
                    #complete_semimasked_U = preU / X_onehot.shape[0]
                        nearPreY = beta_tt[None] * preX  #09/20/25
                        preY[nn*N_batch_size:(nn+1)*N_batch_size] = np.sum(nearPreY, axis=tuple(range(1,len(tup)+1)))  #09/20/25
                elif TENSOR_COMPUTE_STYLE=="N_batching_dataset": 
                    raise NotImplementedError("marginals not implemented for this yet")
                else:
                    raise ValueError(f"TENSOR_COMPUTE_STYLE={TENSOR_COMPUTE_STYLE} not supported")
            # print('complete_semimasked_U')
            # print(complete_semimasked_U)

            ##complete_semimasked_U = None
            ##beta_tt_adjusted = purify_discrete(beta_tt, complete_semimasked_U, anova_purification_type)
            beta_tt_adjusted = beta_tt
            if ASSUME_UNIFORM_DISTRIBUTION:
                pass
                #beta_power_bands[len(tup)] += np.mean(beta_tt_adjusted**2) #wont work for correlated -- maybe remove
            else:
                ##beta_power_bands[len(tup)] += np.sum(complete_semimasked_U*beta_tt_adjusted**2)
                beta_power_bands[len(tup)] += np.mean(preY**2)
            beta_dict[tup] = beta_tt_adjusted

        # print('beta_power_bands')
        # for k in range(self.K):
        #     print('\t',k+1,beta_power_bands[k+1])
        # print()
                
        final_beta_dict = {} #rescaled version
        for tup in all_indices[0:]:
            final_beta_tt = beta_dict[tup] * np.sqrt( band_strengths[len(tup)] / beta_power_bands[len(tup)] )
            final_beta_dict[tup] = final_beta_tt

        self.generator_parameters['final_beta_dict'] = final_beta_dict
        self.generator_parameters['beta_dict'] = beta_dict



    
    def generate_output_samples(self, X_onehot):
        pass
        N = X_onehot[0].shape[0]

        final_beta_dict = self.generator_parameters['final_beta_dict']
        all_indices = self.generator_parameters['all_indices']
        ###I_k = self.I_k
        I_k = 1 #09/20/25
        
        full_y = 0.0

        tup_size = 1
        for tup in all_indices:
            if len(tup) != tup_size:
                tup_size = len(tup)
                print()

            #final_beta_tt = beta_dict[tup] / np.sqrt(beta_power_bands[len(tup)]) / np.sqrt(3.)
            final_beta_tt = final_beta_dict[tup] 
            # print('\t','final_beta_tt',np.mean(np.abs(final_beta_tt)))
            # print('\t','final_beta_tt',np.mean((final_beta_tt**2)))
            # xd=final_beta_tt.reshape(-1)
            # plt.scatter(np.arange(xd.shape[0]),xd); plt.plot(np.arange(xd.shape[0]), xd); plt.ylim(-0.5,0.5); plt.show();

            
            if TENSOR_COMPUTE_STYLE == "ituple_batching_indices":
                preY = np.zeros(X_onehot.shape[0])
                for i_tup in product(*([list(range(I_k))]*len(tup))):
                    print('\t',i_tup)
        
        
                    inds = np.ones(X_onehot.shape[0],dtype=bool)
                    for kk,k in enumerate(tup):
                        #NOTE: this indexing is not general and depends on the [I_k]^d form.
                        inds = np.logical_and(inds, X_onehot[:,I_k*k+i_tup[kk]]==1)
                    preY[inds] += final_beta_tt[i_tup]

            elif TENSOR_COMPUTE_STYLE == "N_batching_dataset":
                #NOTE: NEED TO ADAPTIVELY SELECT "N_BS" to fit on the GPU properly (tuple by tuple in all likelihood)
                preY = np.zeros(X_onehot.shape[0])
                N_batch_size = 100*1000
                
                for nn in range(11):
                    preX = np.ones(())
                    for kk,k in enumerate(tup):
                        preX = preX[...,None] * X_onehot[nn*N_batch_size:(nn+1)*N_batch_size,I_k*k:I_k*(k+1)][ tuple([slice(None)]+[(None)]*(kk)) ]
                        # print('preX',preX.shape)
                    nearPreY = final_beta_tt[None] * preX
                    preY[nn*N_batch_size:(nn+1)*N_batch_size] = np.sum(nearPreY, axis=tuple(range(1,len(tup)+1)))

            else:
                raise ValueError(f"TENSOR_COMPUTE_STYLE={TENSOR_COMPUTE_STYLE} not supported")


            full_y +=  preY
            print('partial_y','var',np.var(full_y))
        print('full_y','var',np.var(full_y))
        return full_y



