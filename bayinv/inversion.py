


import numpy as np
import itertools
from numba import jit
import matplotlib.pyplot as plt
from scipy import sparse 
import matplotlib.colors as mcolors
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import seaborn as sns

plt.style.use("seaborn")
sns.set_style("whitegrid")

@jit(nopython=True)
def Cm_matrix(n, beta):
    Cm = np.zeros((n,n))
    for i, ivalue in enumerate(Cm):
        for j, jvalue in enumerate(ivalue):
            Cm[i,j] = np.exp((-1)*np.abs((i+1)-(j+1)) / (2 * beta)) 
    return Cm

def Cd_inv_matrix(std): 
    return sparse.diags(1 / std)

def G_matrix(combinations, lenght):
    
    dim_row = len(combinations)

    dim_cols = lenght
    
    rows = np.hstack([[i, i]  for i,j in enumerate(combinations)])
    cols = np.hstack(combinations)
    value = np.hstack([1 if i%2 else -1 for i,j in enumerate(cols)])
    
    G = sparse.coo_matrix((value, (rows, cols)), shape=(dim_row,dim_cols), dtype=np.float64)
    
    return G, G.T
    
def plot_data_distribution(dvv):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.histplot(dv/v,  kde=True)
    ax.set_xlabel(xlabel="dvv, %", fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    ax.tick_params(labelsize=18)
    plt.show()
    

def dvv_inversion(dvv, dvv_std,indexes, lenght, beta, alpha):

    dvv_use = dvv[indexes]
    dvv_std_use = dvv_std[indexes]
    combinations = list(itertools.combinations(np.arange(0,lenght), 2))
    combinations = np.vstack(combinations)[indexes]

    
    G, G_T = G_matrix(combinations, lenght)
    
    
    G_Tr = G.T

    Cd_inv = Cd_inv_matrix(dvv_std_use)
    
    Cm = Cm_matrix(lenght, beta)
    
     
    Cm_inv = np.linalg.inv(Cm)
    
    
    del  Cm
    
    R = G_Tr.dot(Cd_inv)
    
    R = R.dot(G)
    
    

    
    R += Cm_inv * alpha

    
    R_inv = np.linalg.inv(R)
    
    m_std = np.diagonal(R_inv)
    
    R_tmp = G.dot(R_inv.T)  
    R = R_tmp.T

    del R_inv, Cm_inv, G_Tr
    
    R = (Cd_inv.T).dot(R_tmp)

    del Cd_inv
    
    R = R.T
    
    m = np.zeros_like(m_std)
    np.dot(R,dvv_use, m)

    
    del R, G

    return m, m_std

def data_filter(dvv, dvv_std, cc_pair, cc_treshold=0.3):
    
    non_nan_dvv_indexes = ~np.isnan(dvv)
    non_nan_std_indexes = ~np.isnan(dvv_std) 
    cc_treshold = cc_pair >=0.3
    index_dvv_std = np.logical_and(non_nan_dvv_indexes, non_nan_std_indexes)
    indexes = np.logical_and(index_dvv_std,cc_treshold)
    
    return indexes


def create_mask(cross_corr_mat, df, dt_minlag):
    
    n_counts = cross_corr_mat.shape[1]
    center_index = np.floor((n_counts - 1.) / 2.)
    left_part = np.arange(0, center_index - df*dt_minlag + 1)
    right_part = np.arange(center_index + df*dt_minlag, n_counts)
    uses_indexes = np.hstack([left_part, right_part]).astype(int)
    mask = np.zeros((n_counts))
    mask[uses_indexes] = 1
    
    return mask
    
def correlation_coefficient_matrix(cross_corr_mat, df, dt_minlag):
    
    mat = cross_corr_mat.copy()
    n = mat.shape[0]
    combinations = list(itertools.combinations(np.arange(0,n), 2))
    cross_corr_number = cross_corr_mat.shape[0]
    combinations_number = len(combinations)
    cc_matrix = np.zeros((cross_corr_number, cross_corr_number))
    cc_array = np.zeros((combinations_number))
    mask = create_mask(mat, df, dt_minlag)
    
    mat *=mask
    
    for i, i_comb in enumerate(combinations):
        n, k = i_comb
        cc_coeff = np.corrcoef(mat[n], mat[k])[0][1]
        
        cc_matrix[n, k] = cc_coeff
        cc_matrix[k, n] = cc_coeff
        cc_array[i] = cc_coeff
        
    for i in range(cross_corr_number):
        cc_matrix[i, i] = 1
    
    return cc_matrix, cc_array, combinations

def plot_cc_matrix(cc_matrix, array_of_date):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    
    
    X, Y = np.meshgrid(array_of_date, array_of_date)
    
    normalize = mcolors.Normalize(vmin=0.3, vmax=1)
    colors = [(1, 1, 224/255), (1, 1, 0), (1, 0., 0.), (0, 0, 0)] 
    cmap = mcolors.LinearSegmentedColormap.from_list('test', colors, N=7)
    
    pcolor = ax.pcolormesh(X, Y, cc_matrix, cmap=cmap, norm=normalize)
    
    ax.set_xlabel("Date", fontsize=20)
    ax.set_ylabel("Date", fontsize=20)
    ax.set_title("CC matrix", fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', rotation=90, labelsize=18)
    ax.invert_yaxis()
    
    cb = fig.colorbar(pcolor, extend='both')
    cb.ax.set_ylabel("Correlation", fontsize=20)
    cb.outline.set_linewidth(2)
    
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(18)
    plt.show()

def plot_dvv_curve(dvv, std, array_of_date, long_term_dvv=None):
    
    fig, ax = plt.subplots(1,1, figsize=(16, 4))

    ax.plot(array_of_date, dvv, "o-", color="black", label="short-term dvv")
    if long_term_dvv is None:
        ax.fill_between(array_of_date, (dvv - std), (dvv + std), color="grey", alpha=0.4)
    else:
        ax.plot(aarray_of_date,  "-", color="blue", label="long-term dvv")
    ax.set_ylabel(ylabel="dv/v, %", fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=14, labelrotation=-90)
    ax.grid(True)
    ax.set_xlabel("Date", fontsize=18)
    ax.legend()
    plt.show()


