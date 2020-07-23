import pandas as pd
import numpy as np
import scipy
import sklearn
import scipy
from scipy import stats
import os
import matplotlib.pyplot as plt
import time


def jupyter_widen():
    """
    Increases width of jupyter cells to use more of the realestate available in the browser
    """
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))

def value_counts(n):
    """
    Wrapper for pandas value_counts for use in groupby
    
    Parameters
    ----------
    n: int
        Number of most common responses
        
    Returns
    -------
    function
        value_counts function that can be used in a groupby
        
    Examples
    --------
    >>> DataFrame.groupby('MEASURE_NAME').agg({'VALUE':['size',value_counts(20)]})
        
    """
    def _value_counts(x):
        return list(x.value_counts().index[0:n])
    _value_counts.__name__='Top {} most common responses'.format(n)
    return _value_counts  

def quantile(n):
    """
    Wrapper for pandas quantile for use in groupby
    
    Parameters
    ----------
    n: int
        Quantile
        
    Returns
    -------
    function
        quantile function that can be used in a groupby
        
    Examples
    --------
    The series on which to apply the returned quantile function must be numeric
    
    >>> DataFrame.groupby('MEASURE_NAME').agg({'VALUE':['size',quantile(0.25)]})
        
    """
    def _quantile(x):
        return np.nanquantile(x, n)
    _quantile.__name__ = '{} quantile'.format(n)
    return _quantile

def _val2col(val):
    """
    Helper function to convert input value into a red hue RGBA
    """
    nonred = abs(0.5-val)*2.0
    return (1.0,nonred,nonred,1.)

def plot_temporal(series_value,series_time,num_bins=20,figpath=None):
    """
    Triplet plot for characterizing longitudinal variables
    
    Parameters
    ----------
    series_value: pandas.Series
        variable value
    series_time: pandas.Series
        variable documentation datetime
    num_bins: int
        number of bins for all subplots
    figpath: str
        path for saving figure
        
    Returns
    -------
    None
        
    Examples
    --------
    >>> plot_temporal(df['VALUE'],df['TIME'],num_bins=30,figpath='./figure.png')
        
    """
    assert pd.api.types.is_numeric_dtype(series_value), 'value series (first parameter) is not numeric'
    assert pd.api.types.is_datetime64_any_dtype(series_time), 'time series (second paramter) is not datetime'
    
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(20,4))
    # histogram - ignore values outside quantile range (0.01,0.99)
    axs[0].hist(series_value,bins=num_bins,range=(series_value.quantile(0.01),series_value.quantile(0.99)))
    
    # number of measurements over time
    axs[1].hist(series_time,bins=num_bins)
    axs[1].tick_params(axis='x', rotation=90)
    
    # rolling quantiles
    quantiles = [.10,.25,.50,.75,.90]
    rollingperc = series_value.groupby(pd.cut(series_time,min(series_value.size,num_bins))).quantile(quantiles).unstack()
    rollingperc = rollingperc.loc[rollingperc.notnull().all(axis=1),:].copy()
    rollingperc.index = pd.IntervalIndex(rollingperc.index)
    # handle such that the last interval.right is added as another row
    newidx = list(rollingperc.index.left)+[rollingperc.index.right[-1]]
    axs[2].fill_between(newidx,list(rollingperc[.10])+[rollingperc[.10].values[-1]],list(rollingperc[.90])+[rollingperc[.90].values[-1]],step='post',color=_val2col(0.1))
    axs[2].fill_between(newidx,list(rollingperc[.25])+[rollingperc[.25].values[-1]],list(rollingperc[.75])+[rollingperc[.75].values[-1]],step='post',color=_val2col(0.25))
    axs[2].step(newidx,list(rollingperc[.50])+[rollingperc[.50].values[-1]],where='post',linestyle='-',color=_val2col(0.5))
    axs[2].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath,dpi=200)
    plt.show()

def dataframe_summary(df,column_item,column_value,stripchars='+-<> '):
    """
    Characterize long dataframe containing longitudinal variables e.g. for mapping purposes
    
    Parameters
    ----------
    df: pandas.DataFrame
        dataframe containing longitudinal variables in long form
    column_item: str
        name of column indicating measurement type
    column_value: int
        name of column indicating measurement result
    stripchars: str
        characters to remove prior to converting to numeric
        
    Returns
    -------
    pandas.DataFrame
        Summary of input dataframe - # of measurements, % numeric, quantiles (0.1,0.25,0.5,0.75,0.9), and top 20 most common results
        
    Examples
    --------
    >>> dataframe_summary(dataframe_laboratoryresults,'LAB_TEST','LAB_RESULT_VALUE')
        
    """
    # helper function for finding what % of measurement results were numeric
    def _notnullperc(x):
        return x.notnull().sum()/x.size*100
    _notnullperc.__name__='% numeric'
    # named size function
    def _size(x):
        return x.size
    _size.__name__='# measurements'
    
    _df = df.copy()
    column_value_numeric = '_num_'+column_value
    _df[column_value_numeric] = pd.to_numeric(_df[column_value].str.strip(stripchars),errors='coerce')
    _ret = _df.groupby(column_item).agg({column_value_numeric:[_size,_notnullperc,quantile(0.1),quantile(0.25),quantile(0.5),quantile(0.75),quantile(0.9)],column_value:[value_counts(20)]})
    _ret.columns = _ret.columns.levels[1]
    return _ret.sort_values(by=['# measurements','% numeric'],ascending=False)

def _name(newname):
    """
    Creates a decorator for naming functions
    """
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

@_name('χ2')
def _chi2(contingency_table):
    """
    Wrapper function for scipy.stats.chi2_contingency
    """
    try:
        return scipy.stats.chi2_contingency(contingency_table)[1]
    except ValueError:
        return 1

@_name('One-way ANOVA')
def _anova(*x):
    """
    Wrapper function for scipy.stats.f_oneway
    """
    return scipy.stats.f_oneway(*x).pvalue

@_name('Kruskal-Wallis H-test')
def _ks(*x):
    """
    Wrapper function for scipy.stats.kruskal
    """
    try:
        return scipy.stats.kruskal(*x).pvalue
    except ValueError as err:
        print('column <{}> contains all same values for all groups'.format(x[0].name))
        return 1.

def cohort_comparison(df,groups,include=[],p_thres=0.01,test_cat=_chi2,test_cont=_ks):
    """
    Generates cohort comparison table
    
    Parameters
    ----------
    df: pandas.DataFrame
        pandas dataframe in the form of [samples x features] where features include group(s) to be compared
    groups: str or list of str
        name(s) of columns to be used as groups for comparison
        if a str, the function will compare those who had True vs. False in the column
    include: list of str
        list of features to be compared. If empty list, all features will be compared.
    p_thres: float
        p-value threshold for significance
    test_cat: function
        statistical test for comparing categorical or boolean variables. Only pre-existing option is _chi2.
    test_cont: function
        statistical test for comparing continuous (numeric) variables. Pre-existing option are _anova and _ks.
    
    Returns
    -------
    pandas.DataFrame
        cohort comparison table
        
    Examples
    --------
    >>> cohortcomparison(processed_dataframe,'In-hospital mortality')
        
    """
    _df = df.copy()

    # if include is empty, then use all columns
    if len(include)==0:
        include = list(_df.columns)
    _df = _df.loc[:,include].copy()

    # groups can be single string (outcome vs. non-outcome) or multiple groups. this means that statistical tests need to be able to handle more-than-2-way comparisons
    if type(groups)==str:
        _df['Non-{}'.format(groups)] = ~_df[groups]
        groups = [groups,'Non-{}'.format(groups)]

    # if any group is in include, remove them from include
    for group in groups:
        if group in include:
            include.remove(group)

    # checking if all requested columns actually exist in the dataframe
    for group in groups+include:
        assert group in _df, 'column ({}) not present in dataframe'.format(group)

    # just making sure all columns are present
    assert all([i in _df.columns for i in include+groups]), 'ERROR: missing group'

    # also add the "total" as a pseudo-group
    _df['Total'] = True
    totgroups = ['Total']+groups

    # categorize columns - datetime or timedelta columns should be transformed into either boolean, number, or category columns
    boolcols = list(_df.select_dtypes(include='bool').columns)
    numcols = list(_df.select_dtypes(include='number').columns)
    catcols = list(_df.select_dtypes(include='category').columns)
    
    # identify which columns will be ignored
    orphancols = [i for i in include if i not in boolcols+numcols+catcols]
    for orphancol in orphancols:
        print('Ignored column: {:30} (dtype: {})'.format(orphancol,_df[orphancol].dtype.name))

    # building the return dataframe as a list of dicts
    _ret=[]

    # Number of samples is a unique case since it's not a "covariate"
    temp_ret={'Variable':'Number of samples, n (%)'}
    for group in totgroups:
        temp_ret[group] = '{:,} ({:.1%})'.format(_df.loc[_df[group]==True,:].shape[0],_df.loc[_df[group]==True,:].shape[0]/_df.shape[0])
        temp_ret['Missing ({})'.format(group)] = '-'
    propchi2= _df.loc[:,groups].agg(['sum','size']).T
    propchi2['non'] = propchi2['size']-propchi2['sum']
    propchi2= propchi2.drop('size',axis=1)
    temp_ret['raw p'] = test_cat(propchi2)
    temp_ret['test'] = test_cat.__name__
    _ret.append(temp_ret)

    for col in [i for i in _df.columns if i in include]:
        
        # BOOLEAN
        if col in boolcols:
            temp_ret={'Variable':'{}, n (%)'.format(col)}
            for group in totgroups:
                temp_ret[group] = '{:,} ({:.1%})'.format(((_df[group]==True) & (_df[col]==True)).sum(), ((_df[group]==True) & (_df[col]==True)).sum() / (_df[group]==True).sum())
                temp_ret['Missing ({})'.format(group)] = '{:,} ({:.1%})'.format(((_df[group]==True) & (_df[col].isnull())).sum(),((_df[group]==True) & (_df[col].isnull())).sum() / (_df[group]==True).sum())
            crosstab = _df.loc[_df[col].notnull(),:].groupby(col)[groups].agg('sum')
            temp_ret['raw p'] = test_cat(crosstab)
            temp_ret['test'] = test_cat.__name__
            _ret.append(temp_ret)
            
        # NUMERIC
        elif col in numcols:
            temp_ret={'Variable':'{}, median (IQR)'.format(col)}
            for group in totgroups:
                temp_ret[group] = '{:,.1f} ({:,.1f} - {:,.1f})'.format(*_df.loc[(_df[group]==True) & (_df[col].notnull()),col].quantile([.5,.25,.75]))
                temp_ret['Missing ({})'.format(group)] = '{:,} ({:.1%})'.format(((_df[group]==True) & (_df[col].isnull())).sum(),((_df[group]==True) & (_df[col].isnull())).sum() / (_df[group]==True).sum())

            temp_ret['raw p'] = test_cont(*[_df.loc[(_df[i]==True) & (_df[col].notnull()),col] for i in groups])
            temp_ret['test'] =  test_cont.__name__
            _ret.append(temp_ret)
            
        # CATEGORICAL
        elif col in catcols:

            #combined
            temp_ret={'Variable':'{}, n (%)'.format(col)}
            for group in totgroups:
                temp_ret[group] = '-'
                temp_ret['Missing ({})'.format(group)] = '{:,} ({:.1%})'.format(((_df[group]==True) & (_df[col].isnull())).sum(),((_df[group]==True) & (_df[col].isnull())).sum() / (_df[group]==True).sum())
            crosstab = _df.groupby([col])[groups].agg('sum')
            temp_ret['raw p'] = test_cat(crosstab)
            temp_ret['test'] = test_cat.__name__
            _ret.append(temp_ret)

            #individual - each individual category compared as if boolean
            dummies = pd.concat([_df.loc[:,totgroups],pd.get_dummies(_df[col])],axis=1)
            for dummycol in _df[col].unique():
                temp_ret={'Variable':'{}, n (%)'.format(dummycol)}
                for group in totgroups:
                    temp_ret[group] = '{:,} ({:.1%})'.format(((dummies[group]==True) & (dummies[dummycol]==True)).sum(), ((dummies[group]==True) & (dummies[dummycol]==True)).sum() / (dummies[group]==True).sum())
                    temp_ret['Missing ({})'.format(group)] = '-'
                crosstab = dummies.loc[dummies[dummycol].notnull(),:].groupby(dummycol)[groups].agg('sum')
                temp_ret['raw p'] = test_cat(crosstab)
                temp_ret['test'] = test_cat.__name__
                _ret.append(temp_ret)

    # column name indicating p value less than threshold
    LTpthres='< {}'.format(p_thres)
    
    _ret = pd.DataFrame(_ret)
    _ret['p'] = _ret['raw p']
    _ret['-log10p'] = -np.log10(_ret['raw p']).round(3)
    _ret['p < {}'.format(p_thres)] = ''
    _ret.loc[_ret['raw p']<p_thres,'p < {}'.format(p_thres)]='*'
    _ret = _ret.drop(['raw p'],axis=1)
    _ret.loc[_ret['p'].astype(float)<p_thres,'p']=LTpthres
    _ret.loc[_ret['p']!=LTpthres,'p'] = _ret.loc[_ret['p']!=LTpthres,'p'].astype(float).apply('{:.3f}'.format)

    
    return _ret

def onehotify(df,sep='|'):
    """
    Wrapper for one-hot encoding all (explicitly) categorical columns in a dataframe
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe with categorical columns to be one-hot encoded. Some pre-processing may be required as this function does not group low-frequency categories.
    sep: str
        Separator. The returned dataframe will contain columns formatted as variable name followed by separator followed by category name.
    
    Returns
    -------
    pandas.DataFrame
        Input dataframe but with categoriacl columns split out into one-hot encoded columns
        
    Examples
    --------
    >>> onehotify(dataframe_demographics)
        
    """
    _df = df.copy()
    catcols = _df.select_dtypes(include='category').columns
    for col in catcols:
        catcolvalcount = _df[col].value_counts()/_df.shape[0]
        temp = (pd.get_dummies(_df[col])==1)
        temp.columns = ['{}{}{}'.format(col,sep,i) for i in temp.columns]
        _df = pd.concat([_df.drop(col,axis=1),temp],axis=1)
    return _df

def boxcox(df,invert=None):
    """
    Forward and inverse boxcox transformation
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe with numeric columns to be boxcox transformed
    invert: dict
        used to perform inverse transformation.
    
    Returns
    -------
    pandas.DataFrame
        boxcox transformed input dataframe
    dict
        contains information regarding forward transformation which can be used to perform inverse transformation
        dict of str (boxcox-transformed column name):dict, which contains as keys 'min' and 'lmbda'
    
    Examples
    --------
    >>> Transformed_DataFrame,Transformation_dict = boxcox(DataFrame)
    >>> Inverse_Transformed_DataFrame = boxcox(Transformed_DataFrame,invert=Transformation_dict)
        
    """
    _df = df.copy()
    if invert is not None:
        for col in invert:
            _df[col] = scipy.special.inv_boxcox(_df[col],invert[col]['lmbda'])
            _df[col] += invert[col]['min']
        return _df
    else:
        transformation={}
        for col in _df.select_dtypes('number'):
            transformation[col]={'min':_df[col].min()-_df[col].std()}
            _df[col]-=transformation[col]['min']
            col_transformed,lmbda = scipy.stats.boxcox(_df.loc[_df[col].notnull(),col])
            transformation[col]['lmbda'] = lmbda
            _df.loc[_df[col].notnull(),col] = col_transformed
        return _df,transformation

def standardize(df,invert=None):
    """
    Forward and inverse standardization transformation (mean=0, std=1)
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe with numeric columns to be standardized
    invert: dict
        used to perform inverse transformation.
    
    Returns
    -------
    pandas.DataFrame
        standardized input dataframe
    dict
        contains information regarding forward transformation which can be used to perform inverse transformation
        dict of str (boxcox-transformed column name):dict, which contains as keys 'std' and 'mean'
    
    Examples
    --------
    >>> Transformed_DataFrame,Transformation_dict = standardize(DataFrame)
    >>> Inverse_Transformed_DataFrame = standardize(Transformed_DataFrame,invert=Transformation_dict)
        
    """
    _df = df.copy()
    if invert is not None:
        for col in invert:
            _df[col]*=invert[col]['std']
            _df[col]+=invert[col]['mean']
        return _df
    else:
        transformation={}
        for col in _df.select_dtypes('number'):
            transformation[col]={'mean':_df[col].mean(),'std':_df[col].std()}
            _df[col] = (_df[col]-transformation[col]['mean'])/transformation[col]['std']
        return _df,transformation

def _charlson():
    """
    Wrapper function for charlson comorbidity dictionary
    """
    return {
        'Myocardial infarction':{
            10:['I21.x', 'I22.x', 'I25.2'],
            9:[ '410.x', '412.x'],
            'Original':1,
            'Quan11':0,
        },
        'Congestive heart failure':{
            10:['I09.9', 'I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5-I42.9', 'I43.x', 'I50.x', 'P29.0'],
            9:['398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93', '425.4-425.9', '428.x'],
            'Original':1,
            'Quan11':2,
        },
        'Peripheral vascular disease':{
            10:['I70.x', 'I71.x', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0', 'I79.2', 'K55.1', 'K55.8', 'K55.9', 'Z95.8', 'Z95.9'],
            9:['093.0', '437.3', '440.x', '441.x', '443.1-443.9', '47.1', '557.1', '557.9', 'V43.4'],
            'Original':1,
            'Quan11':0,
        },
        'Cerebrovascular disease':{
            10:['G45.x', 'G46.x', 'H34.0', 'I60.x-I69.x'],
            9:['362.34', '430.x-438.x'],
            'Original':1,
            'Quan11':0,
        },
        'Dementia':{
            10:['F00.x-F03.x', 'F05.1', 'G30.x', 'G31.1'],
            9:['290.x', '294.1', '331.2'],
            'Original':1,
            'Quan11':2,
        },
        'Chronic pulmonary disease':{
            10:['I27.8', 'I27.9', 'J40.x-J47.x', 'J60.x-J67.x', 'J68.4', 'J70.1', 'J70.3'],
            9:['416.8', '416.9', '490.x-505.x', '506.4', '508.1', '508.8'],
            'Original':1,
            'Quan11':1,
        },
        'Rheumatic disease':{
            10:['M05.x', 'M06.x', 'M31.5', 'M32.x-M34.x', 'M35.1', 'M35.3', 'M36.0'],
            9:['446.5', '710.0-710.4', '714.0-714.2', '714.8', '725.x'],
            'Original':1,
            'Quan11':1,
        },
        'Peptic ulcer disease':{
            10:['K25.x-K28.x'],
            9:['531.x-534.x'],
            'Original':1,
            'Quan11':0,
        },
        'Mild liver disease':{
            10:['B18.x', 'K70.0-K70.3', 'K70.9', 'K71.3-K71.5', 'K71.7', 'K73.x', 'K74.x', 'K76.0', 'K76.2-K76.4', 'K76.8', 'K76.9', 'Z94.4'],
            9:['070.22', '070.23', '070.32', '070.33', '070.44', '070.54', '070.6', '070.9', '570.x', '571.x', '573.3', '573.4', '573.8', '573.9', 'V42.7'],
            'Original':1,
            'Quan11':2,
        },
        'Diabetes without chronic complication':{
            10:['E10.0', 'E10.1', 'E10.6', 'E10.8', 'E10.9', 'E11.0', 'E11.1', 'E11.6', 'E11.8', 'E11.9', 'E12.0', 'E12.1', 'E12.6', 'E12.8', 'E12.9', 'E13.0', 'E13.1', 'E13.6', 'E13.8', 'E13.9', 'E14.0', 'E14.1', 'E14.6', 'E14.8', 'E14.9'],
            9:['250.0-250.3', '250.8', '250.9'],
            'Original':1,
            'Quan11':0,
        },
        'Diabetes with chronic complication':{
            10:['E10.2-E10.5', 'E10.7', 'E11.2-E11.5', 'E11.7', 'E12.2-E12.5', 'E12.7', 'E13.2-E13.5', 'E13.7', 'E14.2-E14.5', 'E14.7'],
            9:['250.4-250.7'],
            'Original':2,
            'Quan11':1,
        },
        'Hemiplegia or paraplegia':{
            10:['G04.1', 'G11.4', 'G80.1', 'G80.2', 'G81.x', 'G82.x', 'G83.0-G83.4', 'G83.9'],
            9:['334.1', '342.x', '343.x', '344.0-344.6', '344.9'],
            'Original':2,
            'Quan11':2,
        },
        'Renal disease':{
            10:['I12.0', 'I13.1', 'N03.2-N03.7', 'N05.2-N05.7', 'N18.x', 'N19.x', 'N25.0', 'Z49.0-Z49.2', 'Z94.0', 'Z99.2'],
            9:['403.01', '403.11', '403.91', '404.02', '404.03', '404.12', '404.13', '404.92', '404.93', '582.x', '583.0-583.7', '585.x', '586.x', '588.0', 'V42.0', 'V45.1', 'V56.x'],
            'Original':2,
            'Quan11':1,
        },
        'Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin':{
            10:['C00.x-C26.x', 'C30.x-C34.x', 'C37.x-C41.x', 'C43.x', 'C45.x-C58.x', 'C60.x-C76.x', 'C81.x-C85.x', 'C88.x', 'C90.x-C97.x'],
            9:['140.x-172.x', '174.x-195.8', '200.x-208.x', '238.6'],
            'Original':2,
            'Quan11':2,
        },
        'Moderate or severe liver disease':{
            10:['I85.0', 'I85.9', 'I86.4', 'I98.2', 'K70.4', 'K71.1', 'K72.1', 'K72.9', 'K76.5', 'K76.6', 'K76.7'],
            9:['456.0-456.2', '572.2-572.8'],
            'Original':3,
            'Quan11':4,
        },
        'Metastatic solid tumor':{
            10:['C77.x-C80.x'],
            9:['196.x-199.x'],
            'Original':6,
            'Quan11':6,
        },
        'AIDS/HIV':{
            10:['B20.x-B22.x', 'B24.x'],
            9:['042.x-044.x'],
            'Original':6,
            'Quan11':4,
        },
    }

def _elixhauser():
    """
    Wrapper function for elixhauser comorbidity dictionary
    """
    return {
        'Congestive heart failure':{
            10:['I09.9', 'I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5-I42.9', 'I43.x', 'I50.x', 'P29.0'],
            9:['398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93', '425.4-425.9', '428.x'],
            'Moore17':9,
            'vanWalraven09':7,
        },
        'Cardiac arrhythmias':{
            10:['I44.1-I44.3', 'I45.6', 'I45.9', 'I47.x-I49.x', 'R00.0', 'R00.1', 'R00.8', 'T82.1', 'Z45.0', 'Z95.0'],
            9:['426.0', '426.13', '426.7', '426.9', '426.10', '426.12', '427.0-427.4', '427.6-427.9', '785.0', '996.01', '996.04', 'V45.0', 'V53.3'],
            'Moore17':0,
            'vanWalraven09':5,
        },
        'Valvular disease':{
            10:['A52.0', 'I05.x-I08.x', 'I09.1', 'I09.8', 'I34.x-I39.x', 'Q23.0-Q23.3', 'Z95.2-Z95.4'],
            9:['093.2', '394.x-397.x', '424.x', '746.3-746.6', 'V42.2', 'V43.3'],
            'Moore17':0,
            'vanWalraven09':-1,
        },
        'Pulmonary circulation disorders':{
            10:['I26.x', 'I27.x', 'I28.0', 'I28.8', 'I28.9'],
            9:['415.0', '415.1', '416.x', '417.0', '417.8', '417.9'],
            'Moore17':6,
            'vanWalraven09':4,
        },
        'Peripheral vascular disorders':{
            10:['I70.x', 'I71.x', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0', 'I79.2', 'K55.1', 'K55.8', 'K55.9', 'Z95.8', 'Z95.9'],
            9:['093.0', '437.3', '440.x', '441.x', '443.1-443.9', '447.1', '557.1', '557.9', 'V43.4'],
            'Moore17':3,
            'vanWalraven09':2,
        },
        'Hypertension, (complicated and uncomplicated)':{
            10:['I10.x','I11.x-I13.x', 'I15.x'],
            9:['401.x','402.x-405.x'],
            'Moore17':-1,
            'vanWalraven09':0,
        },
        'Paralysis':{
            10:['G04.1', 'G11.4', 'G80.1', 'G80.2', 'G81.x', 'G82.x', 'G83.0-G83.4', 'G83.9'],
            9:['334.1', '342.x', '343.x', '344.0-344.6', '344.9'],
            'Moore17':5,
            'vanWalraven09':7,
        },
        'Other neurological disorders':{
            10:['G10.x-G13.x', 'G20.x-G22.x', 'G25.4', 'G25.5', 'G31.2', 'G31.8', 'G31.9', 'G32.x', 'G35.x-G37.x', 'G40.x', 'G41.x', 'G93.1', 'G93.4', 'R47.0', 'R56.x'],
            9:['331.9', '332.0', '332.1', '333.4', '333.5', '333.92', '334.x-335.x', '336.2', '340.x', '341.x', '345.x', '348.1', '348.3', '780.3', '784.3'],
            'Moore17':5,
            'vanWalraven09':6,
        },
        'Chronic pulmonary disease':{
            10:['I27.8', 'I27.9', 'J40.x-J47.x', 'J60.x-J67.x', 'J68.4', 'J70.1', 'J70.3'],
            9:['416.8', '416.9', '490.x -505.x', '506.4', '508.1', '508.8'],
            'Moore17':3,
            'vanWalraven09':3,
        },
        'Diabetes, uncomplicated':{
            10:['E10.0', 'E10.1', 'E10.9', 'E11.0', 'E11.1', 'E11.9', 'E12.0', 'E12.1', 'E12.9', 'E13.0', 'E13.1', 'E13.9', 'E14.0', 'E14.1', 'E14.9'],
            9:['250.0-250.3'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Diabetes, complicated':{
            10:['E10.2-E10.8', 'E11.2-E11.8', 'E12.2-E12.8', 'E13.2-E13.8', 'E14.2-E14.8'],
            9:['250.4-250.9'],
            'Moore17':-3,
            'vanWalraven09':0,
        },
        'Hypothyroidism':{
            10:['E00.x-E03.x', 'E89.0'],
            9:['240.9', '243.x', '244.x', '246.1', '246.8'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Renal failure':{
            10:['I12.0', 'I13.1', 'N18.x', 'N19.x', 'N25.0', 'Z49.0-Z49.2', 'Z94.0', 'Z99.2'],
            9:['403.01', '403.11', '403.91', '404.02', '404.03', '404.12', '404.13', '404.92', '404.93', '585.x', '586.x', '588.0', 'V42.0', 'V45.1', 'V56.x'],
            'Moore17':6,
            'vanWalraven09':5,
        },
        'Liver disease':{
            10:['B18.x', 'I85.x', 'I86.4', 'I98.2', 'K70.x', 'K71.1', 'K71.3-K71.5', 'K71.7', 'K72.x-K74.x', 'K76.0', 'K76.2-K76.9', 'Z94.4'],
            9:['070.22', '070.23', '070.32', '070.33', '070.44', '070.54', '070.6', '070.9', '456.0-456.2', '570.x', '571.x', '572.2-572.8', '573.3', '573.4', '573.8', '573.9', 'V42.7'],
            'Moore17':4,
            'vanWalraven09':11,
        },
        'Peptic ulcer disease excluding bleeding':{
            10:['K25.7', 'K25.9', 'K26.7', 'K26.9', 'K27.7', 'K27.9', 'K28.7', 'K28.9'],
            9:['531.7', '531.9', '532.7', '532.9', '533.7', '533.9', '534.7', '534.9'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'AIDS/HIV':{
            10:['B20.x-B22.x', 'B24.x'],
            9:['042.x-044.x'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Lymphoma':{
            10:['C81.x-C85.x', 'C88.x', 'C96.x', 'C90.0', 'C90.2'],
            9:['200.x-202.x', '203.0', '238.6'],
            'Moore17':6,
            'vanWalraven09':9,
        },
        'Metastatic cancer':{
            10:['C77.x-C80.x'],
            9:['196.x-199.x'],
            'Moore17':14,
            'vanWalraven09':12,
        },
        'Solid tumor without metastasis':{
            10:['C00.x-C26.x', 'C30.x-C34.x', 'C37.x-C41.x', 'C43.x', 'C45.x-C58.x', 'C60.x-C76.x', 'C97.x'],
            9:['140.x-172.x', '174.x-195.x'],
            'Moore17':7,
            'vanWalraven09':4,
        },
        'Rheumatoid arthritis/collagen vascular diseases':{
            10:['L94.0', 'L94.1', 'L94.3', 'M05.x', 'M06.x', 'M08.x', 'M12.0', 'M12.3', 'M30.x', 'M31.0-M31.3', 'M32.x-M35.x', 'M45.x', 'M46.1', 'M46.8', 'M46.9'],
            9:['446.x', '701.0', '710.0-710.4', '710.8', '710.9', '711.2', '714.x', '719.3', '720.x', '725.x', '728.5', '728.89', '729.30'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Coagulopathy':{
            10:['D65-D68.x', 'D69.1', 'D69.3-D69.6'],
            9:['286.x', '287.1', '287.3-287.5'],
            'Moore17':11,
            'vanWalraven09':3,
        },
        'Obesity':{
            10:['E66.x'],
            9:['278.0'],
            'Moore17':-5,
            'vanWalraven09':-4,
        },
        'Weight loss':{
            10:['E40.x-E46.x', 'R63.4', 'R64'],
            9:['260.x-263.x', '783.2', '799.4'],
            'Moore17':9,
            'vanWalraven09':6,
        },
        'Fluid and electrolyte disorders':{
            10:['E22.2', 'E86.x', 'E87.x'],
            9:['253.6', '276.x'],
            'Moore17':11,
            'vanWalraven09':5,
        },
        'Blood loss anemia':{
            10:['D50.0'],
            9:['280.0'],
            'Moore17':-3,
            'vanWalraven09':-2,
        },
        'Deficiency anemia':{
            10:['D50.8', 'D50.9', 'D51.x-D53.x'],
            9:['280.1-280.9', '281.x'],
            'Moore17':-2,
            'vanWalraven09':-2,
        },
        'Alcohol abuse':{
            10:['F10', 'E52', 'G62.1', 'I42.6', 'K29.2', 'K70.0', 'K70.3', 'K70.9', 'T51.x', 'Z50.2', 'Z71.4', 'Z72.1'],
            9:['265.2', '291.1-291.3', '291.5-291.9', '303.0', '303.9', '305.0', '357.5', '425.5', '535.3', '571.0-571.3', '980.x', 'V11.3'],
            'Moore17':-1,
            'vanWalraven09':0,
        },
        'Drug abuse':{
            10:['F11.x-F16.x', 'F18.x', 'F19.x', 'Z71.5', 'Z72.2'],
            9:['292.x', '304.x', '305.2-305.9', 'V65.42'],
            'Moore17':-7,
            'vanWalraven09':-7,
        },
        'Psychoses':{
            10:['F20.x', 'F22.x-F25.x', 'F28.x', 'F29.x', 'F30.2', 'F31.2', 'F31.5'],
            9:['293.8', '295.x', '296.04', '296.14', '296.44', '296.54', '297.x', '298.x'],
            'Moore17':-5,
            'vanWalraven09':0,
        },
        'Depression':{
            10:['F20.4', 'F31.3-F31.5', 'F32.x', 'F33.x', 'F34.1', 'F41.2', 'F43.2'],
            9:['296.2', '296.3', '296.5', '300.4', '309.x', '311'],
            'Moore17':-5,
            'vanWalraven09':-3,
        },
    }

def assign_comorbidities(df,column_code,column_version,columns_id):
    """
    Assign elixhauser/charlson comorbidity and comorbidity scores from diagnosis dataframe
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe containing diagnosis codes
    column_code: str
        name of column containing diagnosis code.
    column_version: int or str
        if int, 9 or 10 indicating ICD version
        if str, name of column containing ICD version (9 or 10)
    columns_id: list of str
        list of names of columns to be used as identifier
    
    Returns
    -------
    df_long: pandas.DataFrame
         long-form dataframe showing the mapping from icd code to comorbidity systems
    df_wide: pandas.DataFrame
        wide-form dataframe showing comorbidities and comorbidity score per identifier
    
    Examples
    --------
    >>> df_diagnosis_long,df_diagnosis_wide = assigncomorbidities(df_diagnosis,'ICD_CODE','ICD_VERSION',['ID'])
    >>> df_diagnosis_long,df_diagnosis_wide = assigncomorbidities(df_diagnosis,'ICD_CODE',9,['MRN','CSN'])
        
    """
    _df = df.copy()
    
    # if all data is of one version, then shift things around such that version refers to a column
    if type(column_version)==int:
        _df['VERSION']=column_version
        column_version='VERSION'
    
    # generate mapping file
    diagmap = _df.loc[:,[column_code,column_version]].drop_duplicates()
    _column_code='{}_SANSDOT'.format(column_code)
    diagmap[_column_code] = diagmap[column_code].str.strip('. ')
    
    
    ComorbiditySystems = {'Elixhauser':_elixhauser(),'Charlson':_charlson()}
    for ComorbiditySystem in ComorbiditySystems:
        for comorbidity in ComorbiditySystems[ComorbiditySystem]:
            curtime=time.time()
            print('Processing: {:>10}, {:>41}...'.format(ComorbiditySystem, comorbidity[:40]),end='')
            idxs=[]
            for version in [9,10]:
                for criteria in ComorbiditySystems[ComorbiditySystem][comorbidity][version]:
                    # assume no dot
                    _criteria = criteria.strip('.')
                    # interval
                    if '-' in _criteria:
                        former = _criteria.split('-')[0]
                        latter = _criteria.split('-')[1]
                        former = former.replace('.x','')
                        latter = latter.replace('.x','~')

                        idx = (diagmap[column_version]==version) & (diagmap[_column_code]>=former) & (diagmap[_column_code]<=latter)
                        idxs.append(idx)

                    # single
                    else:
                        _criteria = _criteria.replace('.x','')
                        idx = (diagmap[column_version]==version) & (diagmap[_column_code].str.startswith(_criteria))
                        idxs.append(idx)
            # merge all indices and find if any applies            
            idx = pd.concat(idxs,axis=1).any(axis=1)

            scores = [i for i in ComorbiditySystems[ComorbiditySystem][comorbidity] if i not in [9,10]]
            newcols = [ComorbiditySystem]+['({}) {}'.format(ComorbiditySystem,i) for i in scores]

            # if the comorbidity or comorbidity score columns do not exist, add them
            for newcol in newcols:
                if newcol not in diagmap:
                    diagmap[newcol]=np.nan

            diagmap.loc[idx,newcols] = (comorbidity,*[ComorbiditySystems[ComorbiditySystem][comorbidity][i] for i in scores])

            print(' Complete! ({:5.1f}s), {:>6,} unique codes found'.format(time.time()-curtime,idx.sum()))
    
    _df = _df.merge(diagmap.drop(_column_code,axis=1),how='left',on=[column_code,column_version])
    
    # handle elixhauser
    elix_onehot = _df.groupby(columns_id+['Elixhauser']).size().unstack()>=1
    elix_onehot.columns = ['({}) {}'.format('Elixhauser',i) for i in elix_onehot.columns]
    elix_score = _df.loc[_df['Elixhauser'].notnull(),columns_id+['Elixhauser']+[i for i in _df if '(Elixhauser) ' in i]].drop_duplicates().drop(['Elixhauser'],axis=1).groupby(columns_id).sum()
    elix_wide = elix_score.merge(elix_onehot,how='outer',left_index=True,right_index=True)
    
    ret = _df.loc[:,columns_id].drop_duplicates().merge(elix_wide,how='left',left_on=columns_id,right_index=True)
    ret.loc[:,elix_onehot.columns] = ret.loc[:,elix_onehot.columns].fillna(False)
    ret.loc[:,elix_score.columns] = ret.loc[:,elix_score.columns].fillna(0)
    
    # handle Charlson
    charlson_onehot = _df.groupby(columns_id+['Charlson']).size().unstack()>=1
    charlson_onehot.columns = ['({}) {}'.format('Charlson',i) for i in charlson_onehot.columns]
    charlson_score = _df.loc[_df['Charlson'].notnull(),columns_id+['Charlson']+[i for i in _df if '(Charlson) ' in i]].drop_duplicates().drop(['Charlson'],axis=1).groupby(columns_id).sum()
    charlson_wide = charlson_score.merge(charlson_onehot,how='outer',left_index=True,right_index=True)
    
    ret = ret.merge(charlson_wide,how='left',left_on=columns_id,right_index=True)
    ret.loc[:,charlson_onehot.columns] = ret.loc[:,charlson_onehot.columns].fillna(False)
    ret.loc[:,charlson_score.columns] = ret.loc[:,charlson_score.columns].fillna(0)
    
    return _df,ret





def plot_roc(y_true,y_score,figpath=None):
    """
    Receiver Operating Curve plot
    
    Parameters
    ----------
    y_true: list-like or pandas.Series
        true y labels
    y_score: list-like or pandas.Series
        predicted probability
    figpath: str
        path for saving figure
        
    Returns
    -------
    None
        
    Examples
    --------
    >>> plot_roc(y_train,y_train_pred)
        
    """
    df_perf = performance_metrics(y_true,y_score)
    fig,axs = plt.subplots(figsize=(8,8))
    axs.plot([0,1],[0,1],linestyle=':',color='grey')
    axs.plot(df_perf['FPR'],df_perf['TPR'],label='AUROC: {:.3f}'.format(sklearn.metrics.roc_auc_score(y_true, y_score)))
    axs.set_xlabel('FPR')
    axs.set_ylabel('TPR')
    axs.legend()
    axs.set_xticks(np.arange(0,1.1,0.1))
    axs.set_yticks(np.arange(0,1.1,0.1))
    axs.set_xlim([-0.005,1.005])
    axs.set_ylim([-0.005,1.005])
    axs.grid()
    
    axs.set_title('AUROC')
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath,dpi=200)
    plt.show()

def plot_prc(y_true,y_score,figpath=None):
    """
    Precision Recall Curve plot
    
    Parameters
    ----------
    y_true: list-like or pandas.Series
        true y labels
    y_score: list-like or pandas.Series
        predicted probability
    figpath: str
        path for saving figure
        
    Returns
    -------
    None
        
    Examples
    --------
    >>> plot_prc(y_train,y_train_pred)
        
    """
    df_perf = performance_metrics(y_true,y_score)
    fig,axs = plt.subplots(figsize=(8,8))
    axs.plot([0,1],[1,0],linestyle=':',color='grey')
    axs.plot(df_perf['Recall'],df_perf['Precision'],label='AUPRC: {:.3f}'.format(sklearn.metrics.average_precision_score(y_true, y_score)))
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')
    axs.legend()
    axs.set_xticks(np.arange(0,1.1,0.1))
    axs.set_yticks(np.arange(0,1.1,0.1))
    axs.set_xlim([-0.005,1.005])
    axs.set_ylim([-0.005,1.005])
    axs.grid()
    
    axs.set_title('AUPRC')
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath,dpi=200)
    plt.show()

def plot_threshold(y_true,y_score,figpath=None):
    """
    Threshold plot
    
    Parameters
    ----------
    y_true: list-like or pandas.Series
        true y labels
    y_score: list-like or pandas.Series
        predicted probability
    figpath: str
        path for saving figure
        
    Returns
    -------
    None
        
    Examples
    --------
    >>> plot_threshold(y_train,y_train_pred)
        
    """
    df_perf = performance_metrics(y_true,y_score)
    fig,axs = plt.subplots(figsize=(8,8))
    
    for metric in ['Recall', 'Specificity', 'Precision', 'F1']:
        _metric = metric
        if _metric=='Recall':
            _metric='Recall aka Sensitivity'
        axs.plot(df_perf.index,df_perf[metric],label=_metric,marker=None,linestyle='-',markersize=1)
        
    axs.axvline(df_perf['F1'].idxmax(),color='black')
    axs.text(df_perf['F1'].idxmax(),1,' Threshold at max F1: {:.3f}'.format(df_perf['F1'].idxmax()),va='top')
    
    axs.set_xlabel('Threshold')
    axs.set_ylabel('Performance')
    axs.legend()
    axs.set_xticks(np.arange(0,1.1,0.1))
    axs.set_yticks(np.arange(0,1.1,0.1))
    axs.set_xlim([-0.005,1.005])
    axs.set_ylim([-0.005,1.005])
    axs.grid()
    
    axs.set_title('Threshold plot')
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath,dpi=200)
    plt.show()

def plot_calibration(y_true,y_score,figpath=None):
    """
    Calibration plot
    
    Parameters
    ----------
    y_true: list-like or pandas.Series
        true y labels
    y_score: list-like or pandas.Series
        predicted probability
    figpath: str
        path for saving figure
        
    Returns
    -------
    None
        
    Examples
    --------
    >>> plot_calibration(y_train,y_train_pred)
        
    """
    df_calibration = pd.DataFrame({'y_true':y_true,'y_score':y_score})
    df_calibration['bin'] = pd.cut(df_calibration['y_score'],np.arange(0,1.1,0.1))
    df_calibration = df_calibration.groupby('bin')['y_true'].agg(['size','mean'])
    df_calibration.index = pd.IntervalIndex(df_calibration.index)
    df_calibration = df_calibration.loc[df_calibration['size']>0,:].copy()
    
    fig,axs = plt.subplots(figsize=(8,8))
    
    axs.bar(df_calibration.index.mid,df_calibration['size'],width=0.08,color='lightgrey')
    twinx = axs.twinx()
    
    axs.set_yscale('log')
    axs.set_xlabel('Predicted probability')
    axs.set_ylabel('Number of patients')
    
    twinx.plot(df_calibration.index.mid,df_calibration['mean'],marker='o',linestyle='-',color='red')
    twinx.plot([0,1],[0,1],color='red',linestyle=':')
    twinx.set_ylabel('Actual probability',color='red')
    twinx.set_xticks(np.arange(0,1.1,0.1))
    twinx.set_yticks(np.arange(0,1.1,0.1))
    twinx.set_xlim([-0.01,1.01])
    twinx.set_ylim([-0.01,1.01])
    twinx.tick_params(axis='y',color='red',labelcolor='red')
    twinx.grid(axis='y',color='pink')

    from matplotlib.ticker import ScalarFormatter
    axs.yaxis.set_major_formatter(ScalarFormatter())

    axs.set_title('Calibration plot')
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath,dpi=200)
    plt.show()