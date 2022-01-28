
import numpy as np
import scipy
import itertools
from numba import jit
import scipy.fftpack as sf

@jit(nopython=True, fastmath=True)
def linear_detrend(data):
  
    new_data = data.copy()
   
    dshape = data.shape
    N = dshape[-1]

    
    A = np.ones((N, 2), np.float64)
    A[:, 0] = np.arange(1, N + 1) * 1.0 / N
    
    coef, resids, rank, s = np.linalg.lstsq(A, data)
    new_data -= np.dot(A, coef)
  
        
    # Put data back in original shape.
    ret = np.reshape(new_data, dshape )
  
    return ret

@jit(nopython=True)
def cosine_taper(npts, p=0.1, freqs=None, flimit=None, halfcosine=True):

    
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)


    
    idx1 = 0
    idx2 = frac - 1
    idx3 = npts - frac
    idx4 = npts - 1


    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1


    cos_win = np.zeros(npts)
    if halfcosine:

        cos_win[idx1:idx2 + 1] = 0.5 * (
            1.0 - np.cos((np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) /
                          (idx2 - idx1))))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = 0.5 * (
            1.0 + np.cos((np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) /
                          (idx4 - idx3))))
    else:
        cos_win[idx1:idx2 + 1] = np.cos(-(
            np.pi / 2.0 * (float(idx2) -
                           np.arange(idx1, idx2 + 1)) / (idx2 - idx1)))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = np.cos((
            np.pi / 2.0 * (float(idx3) -
                           np.arange(idx3, idx4 + 1)) / (idx4 - idx3)))

    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0
    return cos_win


def hanning(M, sym=True):

    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w

def boxcar(M, sym=True):
     return np.ones(M, float)
 

def smooth(x, window='boxcar', half_win=3):

    window_len = 2 * half_win + 1

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = boxcar(window_len).astype('complex')
    else:
        w = hanning(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]
    

def getCoherence(dcs, ds1, ds2):
    n = len(dcs)
    dcs = np.reshape(dcs, n)
    ds1 = np.reshape(ds1, n)
    ds2 = np.reshape(ds2, n)
    coh = np.zeros(n, np.complex128)

    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    valids = valids.T[0]
    coh[valids] = dcs[valids] / ds1[valids] * ds2[valids]
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    

    return coh

@jit(nopython=True)
def nextpow2(x):

    return np.ceil(np.log2(np.abs(x)))


def extract_dt_t(count, current_reference, param, taper, padd, hanningwindow):
    
    freqmin, freqmax, df, tmin, window_length, step = param

    cci = current_reference[0]
    
    cci = linear_detrend(cci)
    
    cri = current_reference[1]
    
    
    cri = linear_detrend(cri)
    
    smoothing_half_win = (len(hanningwindow) - 1)/2
    
    cci *= taper
    cri *= taper
    
    fcur = sf.fft(cci, n=padd)[:padd // 2]
    fref = sf.fft(cri, n=padd)[:padd // 2]
    
    fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
    fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2
    
    fref2 = fref2.astype(np.float64)
    fcur2 = fcur2.astype(np.float64)
    
    X = fref * (fcur.conj())
    if smoothing_half_win != 0:
    
        dcur = np.sqrt(scipy.signal.convolve(fcur2,
                                            hanningwindow,
                                            "same"))
    
        dref = np.sqrt(scipy.signal.convolve(fref2,
                                        hanningwindow,
                                        "same"))

        X = scipy.signal.convolve(X, hanningwindow, "same")
        
    
    else:
        dcur = np.sqrt(fcur2)
        dref = np.sqrt(fref2)
    
    dcs = np.abs(X)
    
 
    freq_vec = sf.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
    index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,freq_vec <= freqmax))
    

    n = len(dcs)
    dcs = np.reshape(dcs, n)
    ds1 = np.reshape(dref, n)
    ds2 = np.reshape(dcur, n)
    coh = np.zeros(n, np.complex128)
    
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    valids = valids.T[0]
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])

    coh[abs(coh) > 1.] = 1. + 0j
    
    mcoh = np.mean(coh[index_range])
    use_coh = coh[index_range]
    use_coh[use_coh >= .99] = .99
        
    

    
    w = 1.0 / (1.0 / (use_coh ** 2) - 1.0)
    
    
    w = np.sqrt(w * np.sqrt(dcs[index_range]))
    w = np.real(w)
    

    v = np.real(freq_vec[index_range]) * 2 * np.pi

    phi = np.angle(X)
    phi[0] = 0.
    
    
    phi = np.unwrap(phi)
    phi = phi[index_range]
       

    
       
    m = np.sum(w*v*phi)/np.sum(w*(v**2)) 
    
    
       
    delta_t = m
    
    
    e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
    
    s2x2 = np.sum(v ** 2 * w ** 2)
    sx2 = np.sum(w * v ** 2)
    e = np.sqrt(e * s2x2 / sx2 ** 2) 
    
    delta_err = e
    delta_mcoh = np.real(mcoh)
    time_axis = tmin+window_length/2.+count*step
    
    return np.hstack([time_axis, delta_t, delta_err, delta_mcoh])

extract_dt_t_vectorize = np.vectorize(extract_dt_t, signature="(),(2,l),(m),(n),(),(t)->(j)") 


def chunck_current_reference(current, reference, maxinds, mininds):
    

    return np.vstack([[current[mininds[i]:maxinds[i]], reference[mininds[i]:maxinds[i]]]  for i, _ in enumerate(maxinds)])
                                   
def mwcs(current, reference, freqmin, freqmax, df, tmin, window_length, window_step,
         hanningwindow, smoothing_half_win=5):


    
   
    
    window_length_samples = np.int32(window_length * df)

    padd = int(2 ** (nextpow2(window_length_samples) + 2))



    count = 0
    taper = cosine_taper(window_length_samples, 0.85)


    maxinds = np.arange(window_length_samples, len(current), int(window_step*df))
    mininds = np.arange(0, len(current) - window_length_samples, int(window_step*df))


    delta_t = np.zeros_like(maxinds,  dtype=np.float64)
    delta_err = np.zeros_like(maxinds, dtype=np.float64)
    delta_mcoh = np.zeros_like(maxinds,  dtype=np.float64)
    time_axis = np.zeros_like(maxinds,  dtype=np.float64)

    cycles = len(maxinds)

    for i in range(cycles):
        minind = mininds[i]
        maxind = maxinds[i]
        cci = current[minind:maxind]
        

        cci = linear_detrend(cci)
        
        cri = reference[minind:maxind]

        cri = linear_detrend(cri)
        

        cci *= taper
        cri *= taper

        fcur = sf.fft(cci, n=padd)[:padd // 2]
        fref = sf.fft(cri, n=padd)[:padd // 2]
    
        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2
        
        fref2 = fref2.astype(np.float64)
        fcur2 = fcur2.astype(np.float64)
        # Calculate the cross-spectrum
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
    
            dcur = np.sqrt(scipy.signal.convolve(fcur2,
                                                hanningwindow,
                                                "same"))
    
            dref = np.sqrt(scipy.signal.convolve(fref2,
                                            hanningwindow,
                                            "same"))
            
      
            X = scipy.signal.convolve(X, hanningwindow, "same")
            
        
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)
        
        dcs = np.abs(X)
    
            # Find the values the frequency range of interest
        freq_vec = sf.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,freq_vec <= freqmax))
    
                                                 

        n = len(dcs)
        dcs = np.reshape(dcs, n)
        ds1 = np.reshape(dref, n)
        ds2 = np.reshape(dcur, n)
        coh = np.zeros(n, np.complex128)
        
        valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
        valids = valids.T[0]
        coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
        #coh[coh > (1.0 + 0j)] = 1.0 + 
    
        coh[abs(coh) > 1.] = 1. + 0j
    
        mcoh = np.mean(coh[index_range])
        use_coh = coh[index_range]
        use_coh[use_coh >= .99] = .99
            

        
       
        
        w = 1.0 / (1.0 / (use_coh ** 2) - 1.0)
        
    
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)
        

        v = np.real(freq_vec[index_range]) * 2 * np.pi

        phi = np.angle(X)
        phi[0] = 0.
        
    
        phi = np.unwrap(phi)
        phi = phi[index_range]
       

       
        m = np.sum(w*v*phi)/np.sum(w*(v**2)) 
        
 
       
        delta_t[i] = m
    
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
    
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2) 
        #delta_err.append(e)
        #delta_mcoh.append(np.real(mcoh))
        #time_axis.append(tmin+window_length/2.+count*step)
        delta_err[i] = e
        delta_mcoh[i] = np.real(mcoh)
        time_axis[i] = tmin+window_length/2.+i*window_step
        
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m
        


    return np.vstack([time_axis, delta_t, delta_err, delta_mcoh])


def dtt(mwcs_result, df=0.1, dtt_lag="static", dtt_sides="both",dtt_maxlag=80, dtt_minlag=15, dtt_v=np.nan, distance=np.nan, dtt_maxdt = 0.5, dtt_mincoh=0.5,dtt_maxerr=0.1):
    tArray = mwcs_result[0]
    dtArray = mwcs_result[1]
    errArray = mwcs_result[2]
    cohArray = mwcs_result[3]
    
    if dtt_lag == "static":
        lmlag = -dtt_minlag
        rmlag = dtt_minlag
    else:
        lmlag = -distance / dtt_v
        rmlag = distance / dtt_v
    lMlag = -dtt_maxlag
    rMlag = dtt_maxlag
    if dtt_sides == "both":
        tindex = np.where(((tArray >= lMlag) & (tArray <= lmlag)) | ((tArray >= rmlag) & (tArray <= rMlag)))[0]
    elif dtt_sides == "left":
        tindex = np.where((tArray >= lMlag) & (tArray <= lmlag))[0]
    else:
        tindex = np.where((tArray >= rmlag) & (tArray <= rMlag))[0]


    cohindex = np.where(cohArray >= dtt_mincoh)[0]
    errindex = np.where(errArray <= dtt_maxerr)[0]
    dtindex = np.where(np.abs(dtArray) <= dtt_maxdt)[0]
   
    index = np.intersect1d(cohindex, errindex)
    index = np.intersect1d(tindex, index)

    
    
    index = np.intersect1d(index, dtindex)
   

    errors = errArray[index]
    w = 1.0 / errors
    w[~np.isfinite(w)] = 1.0
    VecXfilt = tArray[index]
    VecYfilt = dtArray[index]
    if len(VecYfilt) >= 2:
    
        return weighted_linear_regression(VecYfilt, VecXfilt,w )
    else:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
@jit(nopython=True,  fastmath=True)
def weighted_linear_regression(dt_t, time_axis, weight ):
    w_sum = np.sum(weight)
    t_aver = np.sum(weight*time_axis) / w_sum
    dt_average = np.sum(weight*dt_t) / w_sum
    t2_aver = np.sum(weight*(dt_t**2)) / w_sum
    
    m = -np.sum(weight*(time_axis - t_aver)*dt_t) / np.sum(weight*(time_axis-t_aver)**2)
    em = (1 / np.sum(weight*(time_axis-t_aver)**2))**0.5
    a = dt_average - m*t_aver
    ea = (t2_aver * em)**0.5
    
    return np.array([m, a, em, ea])

def dv_v_for_combinations(cross_corr_mat, tmin,  freqmin, freqmax, window_length, window_step,
                          df, dtt_lag="static", dtt_sides="both", 
                          dtt_maxlag=80, dtt_minlag=15, dtt_v=None, 
                          interstation_distance=None, dtt_maxdt = 0.5,
                          dtt_mincoh=0.5,dtt_maxerr=0.1):
    
    n = len(cross_corr_mat)
    comb = list(itertools.combinations(np.arange(0,n), 2))
 

    smoothing_half_win = window_length // 2
    window_len = 2 * smoothing_half_win + 1
    hanningwindow = scipy.signal.windows.hann(window_len).astype(np.float64)
    result = np.array([[], [], [], []]).T

    for i_comb in comb:
        mwcs_result = mwcs(cross_corr_mat[i_comb[1]], cross_corr_mat[i_comb[0]],
        freqmin=freqmin, freqmax=freqmax, df=df, 
        tmin=tmin, window_length=window_length, 
        window_step=window_step, hanningwindow=hanningwindow)
        current_res = dtt(mwcs_result, dtt_minlag=dtt_minlag)
        result = np.vstack([result, current_res])
        
    dvv = result.T[0]*100
    dvv_std = result.T[2]*100
    return dvv, dvv_std


