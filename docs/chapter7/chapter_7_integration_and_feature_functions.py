import sbdynt as sbd
import numpy as np

def run_TNO_integration_for_ML(tno='',clones=2):
    '''
    '''

    if(clones==2):
        find_3_sigma=True
    else:
        find_3_sigma=False

    flag, epoch, sim = sbd.initialize_simulation(planets=['jupiter', 'saturn', 'uranus', 'neptune'],
                          des=tno, clones=clones, find_3_sigma=find_3_sigma)

    sim_2 = sim.copy()

    shortarchive = tno + "-short-archive.bin"
    flag, sim = sbd.run_simulation(sim,tmax=0.5e6,tout=50.,filename=shortarchive,deletefile=True)
    
    longarchive = tno + "-long-archive.bin"
    flag, sim_2 = sbd.run_simulation(sim_2,tmax=10e6,tout=1000.,filename=longarchive,deletefile=True)


    flag, a, ec, inc, node, peri, ma, t = sbd.read_sa_for_sbody(sbody=tno,archivefile=shortarchive,nclones=clones)
    pomega = peri+ node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = sbd.read_sa_by_hash(obj_hash='neptune',archivefile=shortarchive)
    q = a*(1.-ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = sbd.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=shortarchive, nclones=clones)
    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))

    flag, l_a, l_ec, l_inc, l_node, l_peri, l_ma, l_t = sbd.read_sa_for_sbody(sbody=tno,archivefile=longarchive,nclones=clones)
    l_pomega = l_peri+ l_node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = sbd.read_sa_by_hash(obj_hash='neptune',archivefile=longarchive)
    l_q = l_a*(1.-l_ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = sbd.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=longarchive, nclones=clones)
    l_rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    l_phirf = np.arctan2(yr, xr)
    l_tiss = apl/l_a + 2.*np.cos(l_inc)*np.sqrt(l_a/apl*(1.-l_ec*l_ec))



    #set of features to remove from the short or long features:
    short_index_remove = [33,35,37,38,39,40,41,45,47,52,54,97,98,99,100,101,102,103,104,105,106,107,108]
    long_index_remove = [0,2]

    #best fit clone:
    n=0
    short_features = calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])

    short_features = np.delete(short_features,short_index_remove)

    long_features = calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])
    long_features = np.delete(long_features,long_index_remove)

    all_features = np.concatenate((long_features, short_features),axis=0)
    features = np.array([all_features])

    for n in range(1,clones+1):
        short_features = calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
        short_features = np.delete(short_features,short_index_remove)
        long_features = calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])
        long_features = np.delete(long_features,long_index_remove)
        all_features = np.concatenate((long_features, short_features),axis=0)
        features = np.append(features,[all_features],axis=0)
   
    return features, shortarchive, longarchive


def print_TNO_ML_results(pred_class,classes_dictionary,class_probs,clones=2):
    nclas = len(classes_dictionary)
    print("Clone number, most probable class, probability of most probable class, ",end ="")
    for n in range(nclas):
        print("probability of %s," % classes_dictionary[n],end ="")
    print("\n",end ="")
    format_string = "%d, %s, "
    for n in range(nclas-1):
        format_string+="%e, "
    format_string+="%e,\n"
    for n in range(0,clones+1):
        print("%d, %s, %e, " % (n,classes_dictionary[pred_class[n]], class_probs[n][pred_class[n]]),end ="")
        for j in range(nclas):
            print("%e, " % class_probs[n][j] ,end ="")
        print("\n",end ="")



def calc_ML_features(time,a,ec,inc,node,argperi,pomega,q,rh,phirf,tn):
    """
    calculate data features from a time-series
    """

    ########################################################
    ########################################################
    #
    # Very basic time-series data features
    #
    ########################################################
    ########################################################
    
    a_min=np.amin(a)
    e_min=np.amin(ec)
    i_min=np.amin(inc)
    q_min=np.amin(q)
    tn_min=np.amin(tn)

    a_max=np.amax(a)
    e_max=np.amax(ec)
    i_max=np.amax(inc)
    q_max=np.amax(q)
    tn_max=np.amax(tn)


    a_del = a_max - a_min
    e_del = e_max - e_min
    i_del = i_max - i_min
    q_del = q_max - q_min
    tn_del = tn_max - tn_min


    a_mean = np.mean(a)
    e_mean = np.mean(ec)
    i_mean = np.mean(inc)
    q_mean = np.mean(q)
    tn_mean = np.mean(tn)

    a_std = np.std(a)
    e_std = np.std(ec)
    i_std = np.std(inc)
    q_std = np.std(q)
    tn_std = np.std(tn)

    a_std_norm = a_std/a_mean
    q_std_norm = q_std/q_mean

    a_del_norm = a_del/a_mean
    q_del_norm = q_del/q_mean


    #arg peri 0-2pi
    argperi = sbd.arraymod2pi(argperi)
    node = sbd.arraymod2pi(node)
    pomega = sbd.arraymod2pi(pomega)

    argperi_min = np.amin(argperi)
    argperi_max = np.amax(argperi)
    argperi_del = argperi_max - argperi_min
    argperi_mean = np.mean(argperi)
    argperi_std = np.std(argperi)

    #recenter arg peri around 0 and repeat
    argperi_zero = sbd.arraymod2pi0(argperi)
    argperi_min2 = np.amin(argperi_zero)
    argperi_max2 = np.amax(argperi_zero)
    argperi2_del = argperi_max - argperi_min
    argperi_mean2 = np.mean(argperi_zero)
    argperi_std2 = np.std(argperi_zero)

    #take the better values for delta, mean, and standard deviation:
    if(argperi2_del < argperi_del):
        argperi_del = argperi2_del
        argperi_mean = sbd.mod2pi(argperi_mean2)
        argperi_std = argperi_std2

    #calculate time derivatives
    dt = time[1:] - time[:-1] 

    da_dt = a[1:] - a[:-1]
    da_dt = da_dt/dt
    de_dt = ec[1:] - ec[:-1]
    de_dt = de_dt/dt
    di_dt = inc[1:] - inc[:-1]
    de_dt = de_dt/dt
    dq_dt = q[1:] - q[:-1]
    dq_dt = dq_dt/dt

    #unwrap the angles first to be sure we get proper differences
    temp = np.unwrap(argperi)
    dargperi_dt = temp[1:] - temp[:-1]
    dargperi_dt = dargperi_dt/dt

    temp = np.unwrap(node)
    dnode_dt = temp[1:] - temp[:-1]
    dnode_dt = dnode_dt/dt

    temp = np.unwrap(pomega)
    dpomega_dt = temp[1:] - temp[:-1]
    dpomega_dt = dpomega_dt/dt

    adot_min = np.amin(da_dt)
    adot_max = np.amax(da_dt)
    adot_mean = np.mean(da_dt)
    adot_std = np.std(da_dt)
    adot_del = adot_max - adot_min

    edot_min = np.amin(de_dt)
    edot_max = np.amax(de_dt)
    edot_mean = np.mean(de_dt)
    edot_std = np.std(de_dt)
    edot_del = edot_max - edot_min


    idot_min = np.amin(di_dt)
    idot_max = np.amax(di_dt)
    idot_mean = np.mean(di_dt)
    idot_std = np.std(di_dt)
    idot_del = idot_max - idot_min


    nodedot_min = np.amin(dnode_dt)
    nodedot_max = np.amax(dnode_dt)
    nodedot_mean = np.mean(dnode_dt)
    nodedot_std = np.std(dnode_dt)
    nodedot_std_norm = nodedot_std/nodedot_mean
    nodedot_del = nodedot_max - nodedot_min
    nodedot_del_norm = nodedot_del/nodedot_mean


    argperidot_min = np.amin(dargperi_dt)
    argperidot_max = np.amax(dargperi_dt)
    argperidot_mean = np.mean(dargperi_dt)
    argperidot_std = np.std(dargperi_dt)
    argperidot_std_norm = argperidot_std/argperidot_mean
    argperidot_del = argperidot_max - argperidot_min
    argperidot_del_norm = argperidot_del/argperidot_mean


    pomegadot_min = np.amin(dpomega_dt)
    pomegadot_max = np.amax(dpomega_dt)
    pomegadot_mean = np.mean(dpomega_dt)
    pomegadot_std = np.std(dpomega_dt)
    pomegadot_std_norm = pomegadot_std/pomegadot_mean
    pomegadot_del = pomegadot_max - pomegadot_min
    pomegadot_del_norm = pomegadot_del/pomegadot_mean

    qdot_min = np.amin(dq_dt)
    qdot_max = np.amax(dq_dt)
    qdot_mean = np.mean(dq_dt)
    qdot_std = np.std(dq_dt)
    qdot_del = qdot_max - qdot_min



    ########################################################
    ########################################################
    #
    # Rotating Frame data features
    #
    ########################################################
    ########################################################

    phirf = sbd.arraymod2pi(phirf)
    
    #divide heliocentric distance into 10 bins and theta_n
    #into 20 bins
    qmin = np.amin(rh) - 0.01
    Qmax = np.amax(rh) + 0.01
    nrbin = 10.
    nphbin = 20.
    dr = (Qmax - qmin) / nrbin
    dph = (2. * np.pi) / nphbin
    # center on the planet in phi (so when we bin, we will
    # add the max and min bins together since they're really
    # half-bins
    phmin = -dph / 2.

    # radial plus aziumthal binning
    # indexing is radial bin, phi bin: rph_count[rbin,phibin]
    rph_count = np.zeros((int(nrbin), int(nphbin)))
    # radial only binning
    r_count = np.zeros(int(nrbin))

    # for calculating the average sin(ph) and cos(ph)
    # indexing is   sinphbar[rbin,resorder]
    resorder_max = 10
    sinphbar = np.zeros((int(nrbin), resorder_max+1))
    cosphbar = np.zeros((int(nrbin), resorder_max+1))

    # divide into radial and azimuthal bins
    nmax = len(rh)
    for n in range(0, nmax):
        rbin = int(np.floor((rh[n] - qmin) / dr))
        for resorder in range(1,resorder_max+1):
            tcos = np.cos(float(resorder)*phirf[n])
            tsin = np.sin(float(resorder)*phirf[n])
            sinphbar[rbin,resorder]+=tsin
            cosphbar[rbin,resorder]+=tcos
        r_count[rbin]+=1.
        phbin = int(np.floor((phirf[n] - phmin) / dph))
        if (phbin == int(nphbin)):
            phbin = 0
        rph_count[rbin, phbin] += 1

    # perihelion distance bin stats
    nempty = np.zeros(int(nrbin))
    nadjempty = np.zeros(int(nrbin))
    rbinmin = np.zeros(int(nrbin))
    rbinmax = np.zeros(int(nrbin))
    rbinavg = np.zeros(int(nrbin))
    rbinstd = np.zeros(int(nrbin))

    for nr in range(0, int(nrbin)):
        rbinmin[nr] = 1e9
        for resorder in range(1,resorder_max+1):
            sinphbar[nr,resorder] = sinphbar[nr,resorder]/r_count[nr]
            cosphbar[nr,resorder] = cosphbar[nr,resorder]/r_count[nr]
        for n in range(0, int(nphbin)):
            if (rph_count[nr, n] == 0):
                nempty[nr] += 1
            if (rph_count[nr, n] < rbinmin[nr]):
                rbinmin[nr] = rph_count[nr, n]
            if (rph_count[nr, n] > rbinmax[nr]):
                rbinmax[nr] = rph_count[nr, n]
            rbinavg[nr] += rph_count[nr, n]
        rbinavg[nr] = rbinavg[nr] / nphbin

        for n in range(0, int(nphbin)):
            rbinstd[nr] += (rph_count[nr, n] - rbinavg[nr]) * (
                        rph_count[nr, n] - rbinavg[nr])
        if (not (rbinavg[nr] == 0)):
            rbinstd[nr] = np.sqrt(rbinstd[nr] / nphbin) #/ rbinavg[nr]
        else:
            rbinstd[nr] = 0.

        if (rph_count[nr, 0] == 0):
            nadjempty[nr] = 1
            for n in range(1, int(np.floor(nphbin / 2.)) + 1):
                if (rph_count[nr, n] == 0):
                    nadjempty[nr] += 1
                if (rph_count[nr, n] != 0):
                    break
            for n in range(int(nphbin) - 1, int(np.floor(nphbin / 2.)), -1):
                if (rph_count[nr, n] == 0):
                    nadjempty[nr] += 1
                if (rph_count[nr, n] != 0):
                    break


    n_peri_empty = nempty[0]
    n_apo_empty = nempty[-1]
    nadj_peri_empty = nadjempty[-1]
    nadj_apo_empty = nadjempty[-1]

    navg_peri = rbinavg[0]
    nstd_peri = rbinstd[0]
    ndel_peri = rbinmax[0] - rbinmin[0]
    if(navg_peri>0):
        ndel_peri_norm = ndel_peri/navg_peri
        nstd_peri_norm = nstd_peri/navg_peri
    else:
        ndel_peri_norm = 0.
        nstd_peri_norm = 0.

    navg_apo = rbinavg[-1]
    nstd_apo = rbinstd[-1]
    ndel_apo = rbinmax[-1] - rbinmin[-1]
    if(navg_apo>0):
        ndel_apo_norm = ndel_apo/navg_apo
        nstd_apo_norm = nstd_apo/navg_apo
    else:
        ndel_apo_norm = 0.
        nstd_apo_norm = 0.
 
    #add the rayleigh z-test statistics at perihelion and aphelion
    rz_peri = np.zeros(resorder_max+1)
    rz_apo = np.zeros(resorder_max+1)
    for resorder in range(1, resorder_max+1):
        rz_peri[resorder] = np.sqrt(sinphbar[0,resorder]*sinphbar[0,resorder] +
                       cosphbar[0,resorder]*cosphbar[0,resorder])
        rz_apo[resorder] = np.sqrt(sinphbar[-1,resorder]*sinphbar[-1,resorder] +
                       cosphbar[-1,resorder]*cosphbar[-1,resorder])


    rzperi_max = np.amax(rz_peri[1:resorder_max])
    rzapo_max = np.amax(rz_apo[1:resorder_max])


    spatial_counts = rph_count.flatten()
    grid_nz_minval = np.min(spatial_counts[np.nonzero(spatial_counts)])
    grid_nz_avg = np.mean(spatial_counts[np.nonzero(spatial_counts)])
    grid_nz_std = np.std(spatial_counts[np.nonzero(spatial_counts)])
    grid_avg =  np.mean(spatial_counts)
    grid_std =  np.std(spatial_counts)
    grid_deltaavg = grid_nz_avg - grid_avg
    grid_deltastd = grid_std - grid_nz_std
    
    n_empty=0
    n_almost_empty=0
    for n in range(0,len(spatial_counts)):
        if(spatial_counts[n]==0):
            n_empty += 1
        if(spatial_counts[n]<7):
            n_almost_empty += 1

    ########################################################
    ########################################################
    #
    # FFT data features
    #
    ########################################################
    ########################################################

    #calculate the correlations between a and e, a and i, and e and i
    aecorr =  max_corelation(a,ec)
    aicorr =  max_corelation(a,inc)
    eicorr =  max_corelation(ec,inc)

    #calculate spectral fractions
    deltat = time[2] - time[1]
    #a
    asf, amaxpower, amaxpower3, af1, af2, af3 = spectral_characteristics(a,deltat)
    # eccentricity, via e*sin(varpi)
    hec = ec*np.sin(pomega)
    esf, emaxpower, emaxpower3, ef1, ef2, ef3 = spectral_characteristics(hec,deltat)
    # inclination, via sin(i)sin(Omega)
    pinc = np.sin(inc)*np.sin(node)
    isf, imaxpower, imaxpower3, if1, if2, if3 = spectral_characteristics(pinc,deltat)
    #amd
    amd = 1. - np.sqrt(1.- ec*ec)*np.cos(inc)
    amd = amd*np.sqrt(a)
    amdsf, amdmaxpower, amdmaxpower3, amdf1, amdf2, amdf3 = spectral_characteristics(amd,deltat)


    ########################################################
    ########################################################
    #
    # additional time-series based features
    #
    ########################################################
    ########################################################

    #Do some binning in the a, e, and i-distributions
    #compare visit distributions


    em_a, lh_a, min_em_a, max_em_a, delta_em_a, delta_em_a_norm, min_lh_a, max_lh_a, delta_lh_a, delta_lh_a_norm =  histogram_features(a,a_min,a_max,a_mean,a_std)
    em_e, lh_e, min_em_e, max_em_e, delta_em_e, delta_em_e_norm, min_lh_e, max_lh_e, delta_lh_e, delta_lh_e_norm =  histogram_features(ec,e_min,e_max,e_mean,e_std)
    em_i, lh_i, min_em_i, max_em_i, delta_em_i, delta_em_i_norm, min_lh_i, max_lh_i, delta_lh_i, delta_lh_i_norm =  histogram_features(inc,i_min,i_max,i_mean,i_std)
    

    features = [
        a_min,a_mean,a_max,a_std,a_std_norm,a_del,a_del_norm,
        adot_min,adot_mean,adot_max,
        adot_std,adot_del,
        e_min,e_mean,e_max,e_std,e_del,
        edot_min,edot_mean,edot_max,edot_std,edot_del,
        i_min,i_mean,i_max,i_std,i_del,
        idot_min,idot_mean,idot_max,idot_std,idot_del,
        nodedot_min,nodedot_mean,nodedot_max,nodedot_std,
        nodedot_std_norm,nodedot_del,nodedot_del_norm,
        argperi_min,argperi_mean,argperi_max,argperi_std,argperi_del,
        argperidot_min,argperidot_mean,argperidot_max,argperidot_std,argperidot_std_norm,argperidot_del,argperidot_del_norm,
        pomegadot_min,pomegadot_mean,pomegadot_max,pomegadot_std,pomegadot_std_norm,pomegadot_del,pomegadot_del_norm,
        q_min,q_mean,q_max,q_std,q_std_norm,q_del,q_del_norm,
        qdot_min,qdot_mean,qdot_max,qdot_std,qdot_del,
        tn_min,tn_mean,tn_max,tn_std,tn_del,
        n_peri_empty,nadj_peri_empty,
        nstd_peri,ndel_peri,
        rzperi_max,
        n_apo_empty,nadj_apo_empty,nstd_apo,ndel_apo,rzapo_max,
        grid_nz_minval,grid_nz_avg,grid_nz_std,grid_std,grid_deltastd,n_empty,
        aecorr,aicorr,eicorr, 
        asf,amaxpower,amaxpower3,af1,af2,af3,
        esf,emaxpower,emaxpower3,ef1,ef2,ef3,
        isf,imaxpower,imaxpower3,if1,if2,if3,
        amdsf,amdmaxpower,amdmaxpower3,amdf1,amdf2,amdf3,
        em_a,lh_a,delta_em_a,delta_lh_a,
        em_e,lh_e,
        em_i,lh_i,
        ]
   
    return np.array(features)  # make sure features is a numpy array


def max_corelation(d1, d2):
    d1 = (d1 - np.mean(d1)) / (np.std(d1))
    d2 = (d2 - np.mean(d2)) / (np.std(d2))  
    cmax = (np.correlate(d1, d2, 'full')/len(d1)).max()
    return cmax


def spectral_characteristics(data,dt):
    Y = np.fft.rfft(data)
    n = len(data)
    freq = np.fft.rfftfreq(n,d=dt)
    jmax = len(Y)
    Y = Y[1:jmax]
    Y = np.abs(Y)**2.
    arr1 = Y.argsort()    
    sorted_Y = Y[arr1[::-1]]
    sorted_freq = freq[arr1[::-1]]
    f1 = sorted_freq[0]
    f2 = sorted_freq[1]
    f3 = sorted_freq[2]
    ytot = 0.
    for Y in (sorted_Y):
        ytot+=Y
    norm_Y = sorted_Y/ytot
    count=0
    maxnorm_Y = sorted_Y/sorted_Y[0]
    for j in range(0,jmax-1):
        if(maxnorm_Y[j] > 0.05):
            count+=1
    sf = 1.0*count/(jmax-1.)
    maxpower = sorted_Y[0]
    max3 = sorted_Y[0] + sorted_Y[1] + sorted_Y[2]
    return sf, maxpower, max3, f1, f2, f3

def histogram_features(x,xmin,xmax,xmean,xstd):
    x1 = xmin
    x2 = xmean-0.75*xstd
    x3 = xmean-0.375*xstd
    x4 = xmean+0.375*xstd
    x5 = xmean+0.75*xstd
    x6 = xmax
    if(x1 < x2 and x2 < x3 and x4 < x5 and x5 < x6):
        xbins = [x1,x2,x3,x4,x5,x6]
    else:
        dx = (xmax-xmin)/8.
        x2 = xmin + 2.*dx
        x3 = x2 + dx
        x4 = x3 + 2.*dx
        x5 = x4 + dx
        xbins = [x1,x2,x3,x4,x5,x6]
    xcounts, tbins = np.histogram(x,bins=xbins)

    #average ratio of extreme-x density to middle-x density
    if(xcounts[2] == 0):
        xcounts[2] = 1 #avoid a nan
    em_x = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    #ratio of extreme-low-x density to extreme high-x density
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh_x = (xcounts[0]/xcounts[4])

    #repeat across a couple time bins 
    dj = x.size//4
    xcounts, tbins = np.histogram(x[0:dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em1 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh1 = (xcounts[0]/xcounts[4])
 
    xcounts, tbins = np.histogram(x[dj:2*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em2 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh2 = (xcounts[0]/xcounts[4])
    
    xcounts, tbins = np.histogram(x[2*dj:3*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em3 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh3 = (xcounts[0]/xcounts[4])


    xcounts, tbins = np.histogram(x[3*dj:4*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em4 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh4 = (xcounts[0]/xcounts[4])

    min_em_x = min(em1,em2,em3,em4)
    max_em_x = max(em1,em2,em3,em4)

    delta_em_x = max_em_x - min_em_x
    delta_em_x_norm = delta_em_x/em_x    

    min_lh_x = min(lh1,lh2,lh3,lh4)
    max_lh_x = max(lh1,lh2,lh3,lh4)

    delta_lh_x = max_lh_x - min_lh_x
    delta_lh_x_norm = delta_lh_x/lh_x

    return  em_x, lh_x, min_em_x, max_em_x, delta_em_x, delta_em_x_norm, min_lh_x, max_lh_x, delta_lh_x, delta_lh_x_norm


