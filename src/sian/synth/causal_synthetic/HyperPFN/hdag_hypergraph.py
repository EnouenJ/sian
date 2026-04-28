


#lets have an actual class to hold onto the hyperDAG hypergraph because it might get a little unwieldy later
# and then I can easily have all the helper functions right here like drawing and get_body(), get_immoral()


import math

import numpy as np
import torch
import torch.nn as nn


import matplotlib.pyplot as plt
import copy





class HDAG():
    
    def __init__(self, graph_generation_meta_hypers=None, metainit_seed=None):

        np.random.seed(metainit_seed) #NOTE: assumes below equations only use np randomness 
        if graph_generation_meta_hypers is None:
            self.D = 0
            self.nodes = []
            self.hyperedges = []
            self.hyperparents = {}
        else:
            D = graph_generation_meta_hypers["D"]
            self.D = D
            self.nodes = list(range(D))

            self.hyperedges = []
            self.hyperparents = {}

            graph_gen_type = graph_generation_meta_hypers["graph_gen_type"]

            if graph_gen_type=="barabosi_hyper":
                m_list = graph_generation_meta_hypers["m_list"]
                hyedgelist = barabosi_hyper(D, m_list)
            elif graph_gen_type=="erdos_hyper":
                m_list = graph_generation_meta_hypers["m_list"]
                hyedgelist = erdos_hyper(D, m_list)
                #NOTE: currently assuming default top order
                hyedgelist = [(tuple(S[:-1]),S[-1]) for S in hyedgelist]
                print('hyedgelist',hyedgelist)
            else:
                raise NotImplementedError(f"graph_gen_type={graph_gen_type}")


            hyperparents = {}
            for j in range(D):
                hyperparents[j] = []
            for S,j in hyedgelist:
                hyperparents[j].append(S)
            self.hyperedges = hyedgelist
            self.hyperparents = hyperparents



    def plot(self):
        plot_hypergraph(self.D, self.hyperedges)






### GRAPH ONLY STUFF ### (hopefully)


class DirectedHypergraph():
    def __init__(self):
        self.nodes = []
        self.hyperedges = []





point_size = 5
point_size = 10
point_size = 50
my_point_size = point_size 

def plot_graph(D, edgelist):
    mypos = [] #semicircle
    for i in range(D):
        x=np.cos(  2*np.pi*(0.5-i/(D-1)/2)  )
        y=np.sin(  2*np.pi*(0.5-i/(D-1)/2)  )
        mypos.append((x,y))

    for i in range(D):
        x,y=mypos[i]
        plt.scatter(x,y,c="C"+str(i),s=my_point_size)
        
    for edge in edgelist:
        xy1 = mypos[edge[0]]
        xy2 = mypos[edge[1]]
        plt.arrow(xy1[0],xy1[1],(xy2[0]-xy1[0]),(xy2[1]-xy1[1]),
                      head_width=.05,facecolor='k',length_includes_head=True)
                      
    plt.show()



point_size = 5
point_size = 10
point_size = 50
my_point_size = point_size 

plot_central_squares = True
plot_central_squares = False



def plot_hypergraph(D, hyedgelist):
    mypos = [] #semicircle
    for i in range(D):
        x=np.cos(  2*np.pi*(0.5-i/(D-1)/2)  )
        y=np.sin(  2*np.pi*(0.5-i/(D-1)/2)  )
        mypos.append((x,y))

    for i in range(D):
        x,y=mypos[i]
        plt.scatter(x,y,c="C"+str(i),s=my_point_size)

    arrow_dict = {}
    hyedge_central_marker_dict = {}
    for hyedge in hyedgelist:
        undir_hyedge = tuple(sorted(list(hyedge[0]) + [hyedge[1]]))
        hyedge_len = len(undir_hyedge)
        # if len(hyedge[0])==1:
        if hyedge_len==2:
            edge = (hyedge[0][0],hyedge[1])
            xy1 = mypos[edge[0]]
            xy2 = mypos[edge[1]]
            plt.arrow(xy1[0],xy1[1],(xy2[0]-xy1[0]),(xy2[1]-xy1[1]),
                          # head_width=.25,facecolor='k',length_includes_head=True)
                          head_width=.05,facecolor='k',length_includes_head=True)
        else:
            #real hyper edge
            xys = [mypos[s] for s in undir_hyedge]
            xys = np.array(xys)

            xys2 = []
            xys3 = []
            center = np.mean(xys,axis=0)
            for ss,s in enumerate(undir_hyedge):
                xys2.append(xys[ss])
                # ss2 = math.ceil(ss + 1/2 + hyedge_len/2)  % hyedge_len
                # ss3 = math.floor(ss + 1/2 + hyedge_len/2) % hyedge_len 
                ss4 = math.ceil(ss + 1/2 )  % hyedge_len
                ss5 = math.floor(ss + 1/2) % hyedge_len 
                # perc = 0.20
                # perc = 0.10
                perc = 0.05
                opp_wall = 0.5*( xys[ss4] + xys[ss5] )
                opp = center*(1.0-perc) + perc*(opp_wall)
                xys2.append(opp)
                
                perc3 = 0.15
                perc3 = 0.05
                point = xys[ss]
                opp2 = center*(1.0-perc3) + perc3*(point)
                xys3.append(opp2)
            xys2 = np.array(xys2)
            
            ss4 = undir_hyedge.index(hyedge[1])
            poly = plt.Polygon(xys2, facecolor='gray', alpha=1.0, fill=True)
            ax = plt.gca()
            if True: #hope this helps for tight layout
                poly.set_in_layout(False)
            ax.add_patch(poly)

            perc2 = 0.20
            start = center*perc + xys[ss4]*(1-perc)
            length = perc * (xys[ss4]-center)

            arrow_dict[hyedge] = (start,length)
            hyedge_central_marker_dict[hyedge] = xys3
            
    for hedge in hyedgelist:
        if hedge in arrow_dict:
            (start,length) = arrow_dict[hedge] 
            plt.arrow(start[0],start[1],length[0],length[1],
                      head_width=.05,facecolor='k',length_includes_head=True)

    if plot_central_squares:
        for hyedge in hyedgelist:
            if hyedge in hyedge_central_marker_dict:
                undir_hyedge = tuple(sorted(list(hyedge[0]) + [hyedge[1]]))
                xys3 = hyedge_central_marker_dict[hyedge]
    
                for ii,i in enumerate(undir_hyedge):
                    xy=xys3[ii]
                    x,y=xy[0],xy[1]
                    plt.scatter(x,y,c="C"+str(i),s=my_point_size,marker='s')
                      
    plt.show()








def barabosi(d, m, alpha=1.0, a=1.0):
    X = [0]
    in_dict = {0:0}
    out_dict = {0:0}
    edgelist=[]

    for dd in range(1,d):
        # print('dd',dd)
        if len(X)<=m:
            toadd=copy.copy(X)
        else:
            BA_score = [(in_dict[x]**alpha + a) for x in X]
            BA_score = np.array(BA_score) / sum(BA_score)
            # print("X",X)
            # print('BA_score',BA_score)
            toadd = np.random.choice(X, size=m, replace=False, p=BA_score)
            # print('toadd',toadd)
            
        X.append(dd)
        in_dict[dd] = 0
        out_dict[dd] = 0
        for ee in toadd:
            edgelist.append((dd,ee))
            in_dict[ee] += 1
            out_dict[dd] += 1
    # print(edgelist)
    return edgelist


#reverse directionality from typical BA algorithm because of HDAG
#note, jsut one arbitrary choice of extension to the BA algorihtm
def barabosi_hyper(d, m_list, alpha_list=None, a_list=None):
    K = len(m_list)
    if alpha_list is None:
        alpha_list = [1.0]*K
    if a_list is None:
        a_list = [1.0]*K

    
    X = [0]
    in_dict = {0:0}
    out_dict = {0:0}
    edgelist=[]
    hyedgelist=[]
    XX=[]
    in2_dict = {}
    out2_dict = {}

    all_attach_dicts = {}
    for k in range(K):
        all_attach_dicts[k] = {}
    all_attach_dicts[0][(0,)] = 0

    all_cands_dict = {}
    all_cands_dict[0] = [(0,)]
    for k in range(1,K):
        all_cands_dict[k] = []

    for dd in range(1,d):

        to_add_dict = {}
        for k in range(K):
            toadd_k = []
            cands = all_cands_dict[k]

            if len(cands)<=m_list[k]:
                toadd_k = copy.copy(cands)
            else:
                BA_score_k = [(all_attach_dicts[k][x]**alpha_list[k] + a_list[k]) for x in cands]
                BA_score_k = np.array(BA_score_k) / sum(BA_score_k)
                E=len(cands)
                toadd_k = np.random.choice(list(range(E)), size=m_list[k], replace=False, p=BA_score_k)
                toadd_k = [cands[e] for e in toadd_k]
            to_add_dict[k] = toadd_k

        all_cands_dict[0].append( (dd,) )
        all_attach_dicts[0][(dd,)] = 0
        for k in range(K):
            toadd_k = to_add_dict[k]
            for S in toadd_k:
                hyedgelist.append((S,dd))
                undir_hyedge = tuple(list(S) + [dd])
                if k+1<K:
                    all_cands_dict[k+1].append(undir_hyedge)
                    all_attach_dicts[k+1][undir_hyedge] = 0
                all_attach_dicts[k][S] += 1

    return hyedgelist

from itertools import combinations

#using the fixed number of edges version rather than the legitimately probabilistic "p" version
def erdos(d, m):
    edgelist=[]

    cands = list(combinations(list(range(d)), 2))
    # print('cands',cands)

    if len(cands) <= m:
        toadd = copy.copy(cands)
    else:
        E=len(cands)
        toadd = np.random.choice(list(range(E)), size=m, replace=False)
        toadd = [cands[e] for e in toadd]

    edgelist = toadd
    return edgelist

    # assert d*(d-1)/2 >= M

    # ii = 0
    # while ii < m:


#note that "m_list" for erdos_hyper() is global # of edges whereas barabosi_hyper() is local # of edges
def erdos_hyper(d, m_list):
    hyedgelist=[]   
    K = len(m_list)

    for k in range(K):
        cands = list(combinations(list(range(d)), k+2))
        # print('cands',cands)

        if len(cands) <= m_list[k]:
            toadd_k = copy.copy(cands)
        else:
            E=len(cands)
            toadd_k = np.random.choice(list(range(E)), size=m_list[k], replace=False)
            toadd_k = [cands[e] for e in toadd_k]
        hyedgelist.extend( toadd_k )
    return hyedgelist



