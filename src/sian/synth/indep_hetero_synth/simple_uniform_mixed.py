import numpy as np
import matplotlib.pyplot as plt





#TODO: add seed for initialization and for generation (esp w/ noise terms added)




class IndependentHetero_InputDataGenerator():
    def __init__(self, D, discrete_prob, max_I):

        self.D = D
        self.generator_meta_parameters = {
            'D' : D,
            'discrete_prob' : discrete_prob,
            'max_I' : max_I,
        }
        self.generator_parameters = {}
        
        for d in range(D):
            disc_d = np.random.rand() <= discrete_prob

            I_d = 'cts'
            if disc_d:
                I_d = np.random.randint(2,max_I)

            print(d,disc_d,' \t',I_d)

            if not disc_d:
                self.generator_parameters[d] = (disc_d, 'unif_cts')
            else: 
                self.generator_parameters[d] = (disc_d, 'unif_disc', I_d)

    def generate_samples(self, N):

        X_dict = {}
        
        for d in range(self.D):
            disc_d = self.generator_parameters[d][0]
            if disc_d:
                I_d = self.generator_parameters[d][2]
                X_d = np.random.multinomial(1,np.ones(I_d)/I_d,N)
            else:
                X_d = np.random.rand(N)*2-1
            X_dict[d] = X_d
            # print(X_d.shape)

        return X_dict



class LowFrequencyHarmonicFunction_OutputDataFunction():
    def __init__(self, D, max_harmonic_terms):
        pass
        self.D = D
        self.max_harmonic_terms = max_harmonic_terms
        
        # self.fourier_coefficients = np.random.randn(D,max_harmonic_terms,2)
        self.fourier_coefficients = np.random.randn(D,max_harmonic_terms,2)/(1.0+np.arange(max_harmonic_terms)[None,:,None])  #smooth like zeta(2)=pi^2/6
        self.fourier_coefficients[:,0,:] = 0.0
        

    def generate_output_samples(self, X):
        PLOTTING = False
        PLOTTING = True
        
        N = X[0].shape[0]
        output = np.zeros(N)
        for d in range(self.D):
            X_d = X[d]
            if len(X_d.shape)>1:
                I_d = X_d.shape[1]
            else:
                I_d = 1
            if I_d>1:
                X_d = np.matmul(X_d,np.arange(I_d)) / I_d

            cos_ft = np.cos(np.pi*X_d[:,None]*np.arange(self.max_harmonic_terms)[None])
            sin_ft = np.sin(np.pi*X_d[:,None]*np.arange(self.max_harmonic_terms)[None])
            four_ft = np.stack([cos_ft,sin_ft],axis=-1)
            out_d = four_ft * self.fourier_coefficients[d][None]
            out_d = np.sum(np.sum(out_d,axis=-1),axis=-1)

            if PLOTTING:
                plt.scatter(X_d, out_d, c=out_d, cmap='Spectral')
                plt.title(f"d={d}")
                plt.colorbar()
                plt.show()

            output += out_d
        pass
        return output

