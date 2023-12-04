import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        median = np.median(x)
        abs_deviations = np.abs(x - median)
        return np.mean(abs_deviations)
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc= None
        self.scale= None
        try:
                    self.loc=[]
                    self.scale=[]
                    for i in range(features.shape[1]):
                            self.loc.append(np.median(features[:,i]))
                            self.scale.append(np.sum(np.abs(features[:,i]- np.median(features[:,i])))/len(features[:,i]))
        except:
            self.loc = np.median(features)
            n = len(features)
            self.scale = np.sum(np.abs(features-self.loc))/n
    

        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        try:
                answer=[]
                for i in  range(values.shape[1]):
                    answer.append(-np.log(2*self.scale[i]) - abs(values[:,i] - self.loc[i]) / self.scale[i])
                answer=np.column_stack(answer)   
        except:
            answer=None
            answer=-np.log(2*self.scale) - abs(values - self.loc) / self.scale 

        return answer
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))


