import numpy as np
import copy
from scipy.sparse.linalg import eigsh
#from numpy.linalg import multi_dot


#==========================================================================================================
#			Casas Ibarra implementation
#==========================================================================================================

def f_kn(mX0, metaR, metaI) :	# Loop function
    return (mX0/(32*np.pi**2)) * ( (metaR**2 /(-metaR**2 + mX0**2)) * np.log(metaR**2/(mX0**2)) - (metaI**2 /(-metaI**2 + mX0**2)) * np.log(metaI**2/(mX0**2)))
        
def Loop_matrix(m_chi0, U_chi0_dag, metaR, metaI) :# i.e Lambda Matrix .     G already factorized out
    res11 = 0
    res22 = 0
    res33 = 0
    for i in range (3) : 
        res11 += f_kn(m_chi0[0], metaR, metaI) * U_chi0_dag[0, i]**2  
        res22 += f_kn(m_chi0[1], metaR, metaI) * U_chi0_dag[1, i]**2 
        res33 += f_kn(m_chi0[2], metaR, metaI) * U_chi0_dag[2, i]**2 
    return np.array([[res11, 0, 0], [0, res22,0],[0,0,res33]])
        #for k, n in itertools.product( range(3), range(2)):
        
        
        
        
def Casas_Ibarra(parameters):		# output the full G coupling matrix, a dictionary containing all the values enters as parameters

        
        # ========================== Parameters ====================================================================
    try:
        minpar = parameters["MINPAR"]
        kappain = parameters["KAPPAIN"]
        other = parameters["OTHER"]

        lamHEta3 = minpar["lamHEta3Input"]
        lambda4HEta = minpar["lambda4HEtaInput"]
        lambda5HEta = minpar["lambda5HEtaInput"]
        lamEtaSig = minpar["lamEtaSigInput"]
        vSi = minpar["vSiInput"]
        mEta2 = minpar["mEta2Input"]

        Kap11 = kappain["Kap(1,1)"]
        Kap22 = kappain["Kap(2,2)"]
        Kap33 = kappain["Kap(3,3)"]

        v = other["v"]
        
        #PMNS & neutrinos. Assigns the value.
        theta_12 = np.random.uniform(33.68 - 0.70, 33.68 + 0.73)
        theta_23 = np.random.uniform(48.5 - 0.9, 48.5 + 0.7)
        theta_13 = np.random.uniform(8.52 - 0.11, 8.52 + 0.11)
        delta_d = np.random.uniform(177 - 20, 177 + 19)
        delta_m1 = np.random.uniform(0, np.pi)
        delta_m2 = np.random.uniform(0, np.pi)
        delta_m3 = np.random.uniform(0, np.pi)
        m_nu_1 = other["NuM"]
        dm212 = np.random.uniform(6.10e-23, 8.70e-23)
        dm312 = np.random.uniform(2e-21, 2.4e-21)
        
        masses_neutrino_input = []
        Delta_mnu_input = []
        masses_neutrino_input.append(m_nu_1)
        masses_neutrino_input.append(np.sqrt(m_nu_1**2+dm212))
        masses_neutrino_input.append(np.sqrt(m_nu_1**2+dm312))
        m_nu_2 = masses_neutrino_input[1]
        m_nu_3 = masses_neutrino_input[2]


        
        #R-matrix
        #r = Dic_variables_param['r'].val[0]
        Theta1 = np.random.uniform(0, np.pi)
        Theta2 = np.random.uniform(0, np.pi)
        Theta3 = np.random.uniform(0, np.pi)

        

        #print('\n==========\n')
        #print('Kappa = ',ka11, ka23)
        #Eta masses
        metaR = np.sqrt(np.abs(mEta2+0.5*lamEtaSig*vSi**2+0.5*(lamHEta3+lambda4HEta+lambda5HEta)*v**2))
        metaI = np.sqrt(np.abs(mEta2+0.5*lamEtaSig*vSi**2+0.5*(lamHEta3+lambda4HEta-lambda5HEta)*v**2))

    # mH1_tree = l1*0.5*v**2 + l4Sig*0.5*vSi**2 - np.sqrt((2*lHSig*v*vSi)**2 + (l1*v**2 - l4Sig*vSi**2)**2)
    # mH2_tree = l1*0.5*v**2 + l4Sig*0.5*vSi**2 + np.sqrt((2*lHSig*v*vSi)**2 + (l1*v**2 - l4Sig*vSi**2)**2)
        
        # ============================  PMNS ========================================================================
        
        
        U_nu_12 = [[np.cos(theta_12), np.sin(theta_12), 0 ],
        [-np.sin(theta_12), np.cos(theta_12), 0],
        [0,0,1]]
        
        U_nu_13 = [[np.cos(theta_13), 0, np.sin(theta_13)*np.exp(-1j*delta_d)],
        [0, 1, 0],
        [-np.sin(theta_13)*np.exp(1j*delta_d), 0, np.cos(theta_13)]]
        
        U_nu_23 = [[1, 0, 0],
            [0, np.cos(theta_23), np.sin(theta_23) ],
            [0, -np.sin(theta_23), np.cos(theta_23)]]
        
        U_nu_M = [[1, 0, 0],
            [0, np.exp(1j*delta_m1), 0],
            [0, 0, np.exp(1j*delta_m2)]]
        
        
        U_PMNS = np.dot(U_nu_23, np.dot(U_nu_13,np.dot(U_nu_12,U_nu_M)))
        
        
        
        # =========================== D_nu ==========================================================================
        
        D_nu = [[m_nu_1,0,0], [0,m_nu_2, 0], [0,0, m_nu_3]]
        D_nu_demi = [[np.sqrt(m_nu_1),0,0], [0, np.sqrt(m_nu_2), 0], [0,0, np.sqrt(m_nu_3)]]
        
        # =========================== Fermion & scalar masses =======================================================
        
        m_X0=[[np.sqrt(2)*vSi*Kap11,0 , 0],[0, np.sqrt(2)*vSi*Kap22, 0],[0, 0, np.sqrt(2)*vSi*Kap33]]
        
        
        #Python diagonalization : D_chi = U_x^dag Mx U_x With U_x the rotation matrix given by Python
        #So our Uchi0 = U_x^dag, where Uchi0 is the rotation matrix from the paper

        #Ardila comment: can be properly optimized if one uses the eigvalh function from scipy and then takes the sqrt of the eigenvalues.
        #this method already considers purely positive definite eigenvalues and no need of moving to the imaginary column.
        
        #m_phi02, U_scal0_dag = np.linalg.eig(m_scal0)
        m_chi0, U_chi0_real_dag = np.linalg.eigh(m_X0)
        #m_phi0 = np.sqrt(m_phi02)
        
        #enforce positive eigenvalues -> move to imaginary column 
        U_chi0_dag = np.array(U_chi0_real_dag ,dtype=complex)
        
        negative_value_index = []
        for val in m_chi0 : 
            if val<0 : negative_value_index.append(m_chi0.tolist().index(val)) 
            
            
            
        for i in negative_value_index :
            U_chi0_dag[:,i] = 1j*U_chi0_dag[:,i]
            m_chi0[i] = abs(m_chi0[i])
            
            
            
            #One has to be careful with the copy
            #In order to keep the first copy untuched we use copy.copy
        m_chi0sort = copy.copy(m_chi0)
        m_chi0sort.sort()
        m_chi01_LO = m_chi0sort[0]
        m_chi02_LO = m_chi0sort[1]
        m_chi03_LO = m_chi0sort[2]
        
        
        masses_LO = []
        masses_LO.append(m_chi01_LO)
        masses_LO.append(m_chi02_LO)
        masses_LO.append(m_chi03_LO)
            
            
            # ================================ Neutrino loop mass entries (M matrix) =======================================
            
            
            
            #Matrice containing the loop computations
        M_matrix = Loop_matrix(m_chi0, U_chi0_dag, metaR, metaI) 
            
            
            #Diagonalization of a symmetric (complex) matrix
            #H_Mmatrix = np.dot(M_matrix,M_matrix.conj().T)
            #hm_matrix, UHm_matrix = np.linalg.eigh(H_Mmatrix)
            
            
            #D_M_complex =  np.dot( UHm_matrix.conj().T,np.dot(M_matrix,UHm_matrix.conj()))
            #phase1 = cmath.phase(D_M_complex[0][0])
            #phase2 = cmath.phase(D_M_complex[1][1])
            
            
            #D_phase_demi = np.array([[ cmath.exp(-1j*phase1*0.5)  ,0],[0,  cmath.exp(-1j*phase2*0.5)  ]])
            #U_M = np.dot(UHm_matrix.conj(),D_phase_demi)
            
            
        m_M , U_M = np.linalg.eigh(M_matrix)
            #m_M
            #m_M.append(math.sqrt(abs(hm_matrix[0])))
            #m_M.append(math.sqrt(abs(hm_matrix[1])))
            
        m_M = sorted(np.abs(m_M))
            
            
            #negative_value_index = []
            #zero_value_index = []
            #for val in m_M : 
            #	if val<0 : 
            #		negative_value_index.append(m_M.tolist().index(val)) 
            #	if val==0 : 
            #		zero_value_index.append(m_M.tolist().index(val)) 
            
            
            
            #for i in negative_value_index :
            #	U_M[:,i] = 1j*U_M[:,i]
            #	m_M[i] = abs(m_M[i])
            
            #for i in zero_value_index :
                #m_M[i] = abs(m_M[i])			
                #m_M[i] = abs(m_M[i]+ 1.0e-30)
            
        #print('m_M[] = ', m_M)
            
            #D_M_demi = [[1/math.sqrt(m_M[0]+ 1.0e-10), 0], [0, 1/math.sqrt(m_M[1]+ 1.0e-10)]]
            #D_M_demi = [[1/math.sqrt(m_M[0]), 0], [0, 1/math.sqrt(m_M[1])]]
        Lambda_sqrt = [[1/np.sqrt(m_M[0]), 0,0], [0, 1/np.sqrt(m_M[1]),0],[0,0,1/np.sqrt(m_M[2])]]
            
            # ================================ R-matrix =====================================================================
            
            #R_matrix = [[0, math.cos(alpha), math.sin(alpha)* cmath.exp(-1j*delta_R)], [0, - math.sin(alpha)* cmath.exp(1j*delta_R), math.cos(alpha)]] #Complex orthogonal matrix
            
        s1 = np.sin(Theta1)
        s2 = np.sin(Theta2)
        s3 = np.sin(Theta3)
        
        c1 = np.sqrt(1-s1**2)
        c2 = np.sqrt(1-s2**2)
        c3 = np.sqrt(1-s3**2)
        
        
        R_matrix = [[c2*c3, -s3*c1-s1*s2*s3, s1*s3-c1*s2*c3], [c2*s3, c1*c3-s1*s2*s3 , -s1*c3-c1*s2*s3],[s2, s1*c2, c1*c2]] 
        
        
        # =================================== Output G-matrix ===========================================================
        
        Yn = np.dot(Lambda_sqrt,np.dot(R_matrix,np.dot(D_nu_demi, U_PMNS.conj())))
        Matrix_mass_nu = np.dot( Yn.T, np.dot(M_matrix,Yn) )
        
        #Internal Check of the neutrino masses: 
        H = np.dot(Matrix_mass_nu,Matrix_mass_nu.conj().T)
        hn = np.linalg.eigvalsh(H)
        #hn, Upmns_h = np.linalg.eigh(H)

        hn1 = hn[0]
        hn2 = hn[1]
        hn3 = hn[2]
        if Yn.shape != (3, 3):
            Yn = np.zeros((3, 3))
        matriz_real = np.real(Yn)
        matriz_imag = np.imag(Yn)

        diccionario = {"COUPLINGSYNIN": {
            "Yn(1,1)" : matriz_real[0,0],
            "Yn(1,2)" : matriz_real[0,1],
            "Yn(1,3)" : matriz_real[0,2],
            "Yn(2,1)" : matriz_real[1,0],
            "Yn(2,2)" : matriz_real[1,1],
            "Yn(2,3)" : matriz_real[1,2],
            "Yn(3,1)" : matriz_real[2,0],
            "Yn(3,2)" : matriz_real[2,1],
            "Yn(3,3)" : matriz_real[2,2]
            },
            "IMCOUPLINGSYNIN": {
            "IYn(1,1)" : matriz_imag[0,0],
            "IYn(1,2)" : matriz_imag[0,1],
            "IYn(1,3)" : matriz_imag[0,2],
            "IYn(2,1)" : matriz_imag[1,0],
            "IYn(2,2)" : matriz_imag[1,1],
            "IYn(2,3)" : matriz_imag[1,2],
            "IYn(3,1)" : matriz_imag[2,0],
            "IYn(3,2)" : matriz_imag[2,1],
            "IYn(3,3)" : matriz_imag[2,2]
        }}

   # Test_array = np.array([[1,2,3],[0,1,3]])
   # print('The Yukawa Matrix Yn = \n')
   # print(Yn)
   # print('\n Test ARRAY')
   # print(Test_array)
   # print('The first element of Y = ', Yn[0,0].real)
   # print('The first element of Test array = ',Test_array[0,0])
        
        return diccionario
    except Exception as e:
        Yn = np.zeros((3, 3))

        matriz_real = np.real(Yn)
        matriz_imag = np.imag(Yn)

        diccionario = {"COUPLINGSYNIN": {
            "Yn(1,1)" : matriz_real[0,0],
            "Yn(1,2)" : matriz_real[0,1],
            "Yn(1,3)" : matriz_real[0,2],
            "Yn(2,1)" : matriz_real[1,0],
            "Yn(2,2)" : matriz_real[1,1],
            "Yn(2,3)" : matriz_real[1,2],
            "Yn(3,1)" : matriz_real[2,0],
            "Yn(3,2)" : matriz_real[2,1],
            "Yn(3,3)" : matriz_real[2,2]
            },
            "IMCOUPLINGSYNIN": {
            "IYn(1,1)" : matriz_imag[0,0],
            "IYn(1,2)" : matriz_imag[0,1],
            "IYn(1,3)" : matriz_imag[0,2],
            "IYn(2,1)" : matriz_imag[1,0],
            "IYn(2,2)" : matriz_imag[1,1],
            "IYn(2,3)" : matriz_imag[1,2],
            "IYn(3,1)" : matriz_imag[2,0],
            "IYn(3,2)" : matriz_imag[2,1],
            "IYn(3,3)" : matriz_imag[2,2]
        }}
        return diccionario
        
