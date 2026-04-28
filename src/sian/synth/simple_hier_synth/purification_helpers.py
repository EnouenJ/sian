






import copy
import numpy as np






from sian import powerset 
from sian import get_heredity_scores



























# marginal; conditional;
def purify_one_step_discrete(beta, base_measure_U, purification_style, return_residuals=False): #NOTE: maybe rename residuals to projections
    
    K = len(beta.shape)
    currlen = K 
    purepower = list(powerset(list(range(K))))

    residual = 0.0
    if return_residuals: #NOTE: not the full separated residuals, but only the first steps projections
        residuals_projections_dict = {}

    for purelen in range(K-1,-1,-1): 
        for pp,puretup in enumerate(purepower):
            if len(puretup)==purelen:
                
                #axes_to_sum = tuple( [ii for ii,i in enumerate(tup) if i in puretup] )
                #axes_to_not_sum = tuple( [ii for ii,i in enumerate(tup) if i not in puretup] )
                axes_to_sum = tuple( [ii for ii in range(K) if ii in puretup] )
                axes_to_not_sum = tuple( [ii for ii in range(K) if ii not in puretup] )
                
                if purification_style=="marginal_purification":
                    unflattened_U = np.sum(base_measure_U, axis=axes_to_sum, keepdims=True)
                    unflattened_beta = np.sum(beta, axis=axes_to_not_sum, keepdims=True)
                    beta_pp = unflattened_beta * unflattened_U
                elif purification_style=="conditional_purification":
                    new_unflattened_beta = np.sum(beta * base_measure_U, axis=axes_to_not_sum, keepdims=True)
                    new_unflattened_U = np.sum(base_measure_U, axis=axes_to_not_sum, keepdims=True)
                    beta_pp = new_unflattened_beta / new_unflattened_U
                else:
                    raise ValueError(f"purification_style={purification_style} not supported.")

                sign = 0.0
                if (purelen-currlen)%2 == 1:
                    sign = 1.0
                else:
                    sign = -1.0

                residual = residual + sign * beta_pp
                # print('\t\t\tresiduals',puretup,beta_pp) #02/03/26 @ 6:00pm -- off at 9:30pm
                # print(puretup, sign * beta_pp)
                if return_residuals:
                    residuals_dict[puretup] = sign * beta_pp
                
    beta_adjusted = beta - residual    
    if return_residuals:
        return beta_adjusted, residuals_dict
    return beta_adjusted


# marginal; (conditional) Sobol; (conditional) Hooker
def purify_discrete(beta, base_measure_U, purification_type, return_residuals=False):
    if return_residuals:
        #although the individual projection ones are implemmented, the way to aggregate is not.
        raise NotImplementedError("sorry no downwards residuals yet") #TODO: still not the full downwards versions
    
    if purification_type=="unpurified_ANOVA":
        beta_adj = (beta)
        
    elif purification_type=="marginal_ANOVA":
        beta_adj = purify_one_step_discrete(beta, base_measure_U, "marginal_purification", return_residuals=return_residuals)
    
    elif purification_type=="conditional_Sobol_ANOVA":
        beta_adj = purify_one_step_discrete(beta, base_measure_U, "conditional_purification", return_residuals=return_residuals)
    
    elif purification_type=="conditional_Hooker_ANOVA":
        #TODO: use an adaptive "P_rounds" which will do more when heavily correlated and less when lightly correlated (just measure difference each round)
        P_rounds = 100 #default rounds of purification
        curr_beta = beta
        for pp in range(P_rounds):
            curr_beta = purify_one_step_discrete(curr_beta, base_measure_U, "conditional_purification", return_residuals=return_residuals)        
        beta_adj = curr_beta

    elif purification_type=="purification_off":      
        beta_adj = beta
    
    else:
        raise ValueError(f"ANOVA purification_type={purification_type} not supported.")

    return beta_adj




def shuffled_subset(tuplelist, subset_size):
    perm = np.random.permutation(len(tuplelist))
    return [tuplelist[pp] for pp in list(perm[:subset_size])]

def strongest_heredity_subset(tuplelist, prevtuplelist, subset_size):
    sorted_candidates = []
    # print('tuplelist',tuplelist)
    # print('prevtuplelist',prevtuplelist)

    if subset_size < 1:
        return sorted_candidates

    score_dict = get_heredity_scores(tuplelist, prevtuplelist)
    possible_scores = []
    for cand in tuplelist:
        score = score_dict[cand]
        if score not in possible_scores:
            possible_scores.append(score)
    print('possible_scores',possible_scores)
    print('score_dict',score_dict)

    for score in sorted(possible_scores, reverse=True):
        candidates_sc = [cand for cand in tuplelist if score_dict[cand] == score]
        perm = np.random.permutation(len(candidates_sc))
        sorted_candidates.extend(  [candidates_sc[pp] for pp in list(perm)]  )
        print('sorted_candidates',sorted_candidates)
        if len(sorted_candidates) >= subset_size:
            break
    
    return sorted_candidates[:subset_size]