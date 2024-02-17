from commons import *
from scipy.stats import entropy

# TODO: 检查相同元素的数据，然后 raise ValueError

def normalize_vec(vec, method='prop', low_is_better=False):
    is_legitimate_vec = type(vec) in [np.ndarray, pd.Series]
    assert is_legitimate_vec and (len(vec.shape) == 1), \
        "vec format should be `pandas.Series` or 1d `numpy.ndarray`"
    
    if method == 'prop': # proportional
        summed_score = np.sum(vec)
        normalized_vec = vec / summed_score
    elif method == 'sub': # subtraction
        min_element, max_element = np.min(vec), np.max(vec)
        difference = np.max(vec) - min_element
        if not low_is_better:
            normalized_vec = (vec - min_element) / difference
        else:
            normalized_vec = (max_element - vec) / difference
    else:
        raise NotImplementedError
    
    return normalized_vec

def normalize_mat(data, method='prop', low_is_better=None):
    assert type(data) == pd.DataFrame, \
        "data format should be `pandas.DataFrame` !"
    num_col = data.shape[1]
    if low_is_better == None:
        low_is_better = [False] * num_col
    
    normalized_data = pd.DataFrame({})
    for i_col in range(num_col):
        vec = data.iloc[:,i_col]
        normalized_vec = normalize_vec(vec, method, low_is_better[i_col])
        
        new_series_name = vec.name
        # new_series_name = 'normalized_' + vec.name
        normalized_data[new_series_name] = normalized_vec
    
    return normalized_data

def weight_ROC(data=None, num_property=None, 
        preference=None, method='inverse'):
    """
    Note that the 0-th element of `preference` should be the element
        with top priority!
    """
    data_exist = data is not None
    num_property_exist = num_property is not None
    preference_indicated = preference is not None
    
    if not num_property_exist:
        if data_exist:
            num_property = data.shape[1]
        else:
            raise ValueError
    if not preference_indicated:
        preference = data.columns
    
    original_array = np.arange(1, num_property+1)
    if method == "inverse":
        inversed_array = 1 / original_array
        weights = [
            inversed_array[i-1] / np.sum(inversed_array) \
            for i in range(1, num_property+1)
        ]
    elif method == "addition": 
        weights = [
            (num_property + 1 - i) / np.sum(original_array) \
            for i in range(1, num_property+1)
        ]
    elif method == "exponent": 
        z = np.log(100) / np.log(num_property)
        weights = [
            (num_property + 1 - i)**z / np.sum(original_array**z) \
            for i in range(1, num_property+1)
        ]
    elif method == "simple":
        inversed_array = 1 / original_array
        weights = [
            np.sum(inversed_array[i-1:]) / num_property \
            for i in range(1, num_property+1)
        ]
    weights = pd.Series(weights, index=preference)
    
    return weights

def weight_entropy(normalized_data):
    k = 1 / np.log(normalized_data.shape[0])
    # information entropy
    entropy_vec = k * entropy(normalized_data, axis=0)
    # information utility
    redundancy_vec = 1 - entropy_vec
    
    weights = pd.Series(
       normalize_vec(redundancy_vec), index=normalized_data.columns
    )
    entropy_series = pd.Series(entropy_vec, index=normalized_data.columns)
    
    # breakpoint()
    return weights, entropy_series

def compute_scores(normalized_data, weights):
    assert type(weights) is pd.Series, \
        "`weights` is not `pandas.Series!"
    assert type(normalized_data) is pd.DataFrame, \
        "`normalized_data` is not `pandas.DataFrame!"
    
    scores = pd.DataFrame({})
    for property_name in normalized_data.columns:
        scores[property_name] = \
            normalized_data[property_name] * weights.loc[property_name]
    
    total_score = scores.sum(axis=1)
    scores['TotalScore'] = total_score

    return scores

def compute_rank(scores, ascending=False):
    assert type(scores) is pd.Series, "type of `scores` should be `pd.Series`!"
    ranking = scores.rank(ascending=False).round()
    ranking = ranking.astype('int64')
    return ranking

class Property(object):
    
    def __init__(self):
        
        # self.name = ...
        # self.low_is_better = ...
        pass
        
    def __str__(self):
        
        pass

def establish_property(todo):
    """
    Given a pair or pairs of property name and low_is_better value(s),
        create property instance(s)
    """
    pass

class Analysis(object):
    # TODO: finish
    
    def __init__(self, data, normalized=False, **kwargs):
        
        # TODO: kwargs -> norm_method, low_is_better
        
        if normalized:
            self.data = data
        else:
            raise NotImplementedError
            self.data = normalize_mat(
                data, method=norm_method, 
                low_is_better=low_is_better
            )
    
    # def __str__(self): pass
    
    def get_weight(self, method, **kwargs):
        # TODO: unfold kwargs
        
        method = method.upper()
        assert method.upper() in ['ROC', 'EWM'], \
            'Please use string `ROC` or `EWM` as the method argument!'
        
        if method == 'ROC':
            num_property = self.data.shape[1]
            weights = weight_ROC(self.data)
        elif method == 'EWM':
            pass
        else:
            raise NotImplementedError
        
        return weights
    
    def roc(self):
        pass
    
    def ewm(self):
        pass
