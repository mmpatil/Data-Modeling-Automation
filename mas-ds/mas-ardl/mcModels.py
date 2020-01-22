#!/usr/bin/env python
"""
TODO: Document the module.
Provides classes and functionality for SOME_PURPOSE
"""

#######################################
# Any needed from __future__ imports  #
# Create an "__all__" list to support #
#   "from module import member" use   #
#######################################

__all__ = [
    # Constants
    # Exceptions
    # Functions
        """
        * extract_data
        * regress_data
        * plot_to_excel
        * coef_plot
        * regress_df
        * regress_excel
        * level_change
        * log_change
        * percent_change
        * build_derivatives
        * mev_preprocess
        * pull_scenario
        * univariate_analysis
        * outlier_z_test
        * find_mev
        * check_tables
        * summStats
        """
    # ABC "interface" classes
    # ABC abstract classes
    # Concrete classes
]

#######################################
# Module metadata/dunder-names        #
#######################################

__author__ = 'Michael J. McLaughlin'
__copyright__ = 'Copyright 2018, all rights reserved'
__status__ = 'Development'

#######################################
# Standard library imports needed     #
#######################################

# Uncomment this if there are abstract classes or "interfaces" 
#   defined in the module...
# import abc
import subprocess
import os

#######################################
# Third-party imports needed          #
#######################################
from sklearn import linear_model
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xlwt import Workbook
from PIL import Image
import datetime as dt
from scipy import stats

#######################################
# Local imports needed                #
#######################################

#######################################
# Initialization needed before member #
#   definition can take place         #
#######################################

#######################################
# Module-level Constants              #
#######################################

#######################################
# Custom Exceptions                   #
#######################################

#######################################
# Module functions                    #
#######################################
def extract_data(filename, sheetname):
    """
    Import development data from excel file where the first column is dates, the second column is
    the dependent and the remaining columns are the independent variables
    Inputs: 
    * filename: String
    * sheetname: string
    Outputs:
    * dates: pandas series
    * dependent: pandas series
    * independent: pandas data frame
    """
    data = pd.read_excel(filename, sheet_name = sheetname)

    # Define Dependent and Independent Variables
    independent = data.iloc[:,2:len(data.columns)]
    dependent   = data.iloc[:,1]
    dates       = data.iloc[:,0]
    return dates, dependent, independent


def regress_data(dependent, independent, intercept = True):
    """
    Perform ordinary least squares regression on the data
    inputs:
    * dependent: pandas data series
    * independent: pandas data frame
    * Intercept: defaults to true but set as false if no intercept is desired
    Outputs:
    Results has the following important attributes and methods
    * .summary() - shows a summary of the regression results
    * .params - is a pandas series of the parameters fitted by the model
    """
    if intercept == True:
        X = sm.add_constant(independent)
    results = sm.OLS(dependent, X).fit()
    return results


def plot_to_excel(plotname, workbook, worksheet, num):
    """
    Take a plot that is saved in the files and places it into a specified workbook and worksheet in 
    a position on a nx2 grid of plots
    inputs:
    * plotname - name of the plot as it sits in the files (eg. 'plot.jpg')
    * workbook - name of the variable that has the workbook defined in python, initialized by w = Workbook() 
    * worksheet - name of the worksheet within the workbook defined in python, initialized as ws = w.add_sheet(SHEETNAME)
    * num - number of the plot on the grid moving left to right up to down 
        1  2
        3  4
        5  6 etc..
    outputs: no defined output, the workbook and worksheet passed are edited and ready for save.
    """
    img = Image.open(plotname)
    file_out = plotname[0:-4] + '.bmp'
    if len(img.split()) == 4:
        # prevent IOError: cannot write mode RGBA as BMP
        r, g, b, a = img.split()
        img = Image.merge("RGB", (r, g, b))
        img.save(file_out)
    else:
        img.save(file_out)

    j = np.floor(num/2)
    if np.remainder(num,2) == 0:
        k = 0
    else:
        k = 9
    
    worksheet.insert_bitmap(file_out, int(20*j), int(k))


def coef_plot(results, out):
    """
    Plot the OLS results and return an excel file named by the string with a grid of pictures of the plots
    
    inputs:
    * results - ols results that you would like to plot coefficient diagram for
    * output file name desired (eg. 'coefficients.jpg')
    outputs: no outputs passed back from function, saves the figure in the working directory
    """
    plt.close()
    err_series = results.params - results.conf_int()[0]
    coef_df = pd.DataFrame({'coef': results.params.values[1:],
                            'err': err_series.values[1:],
                            'varname': err_series.index.values[1:]
                           })
    fig, ax = plt.subplots(figsize=(8, 5))
   
    coef_df.plot(x='varname', y='coef', kind='bar', 
                 ax=ax, color='none', 
                 yerr='err', legend=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
               marker='s', s=120, 
               y=coef_df['coef'], color='green')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
    ax.xaxis.set_ticks_position('none')
    _ = ax.set_xticklabels(coef_df['varname'], 
                           rotation=30, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(out) 
    


def regress_df(df, out):
    """
    Use a data frame and return results of the regression performed on it and name the excel file with the string
    inputs:
    * df -  name of data frame containing the dates, dependent variable, and independent variables as columns, in that order
    * out -  name of the output excel file desired, will be used for both _plots.xls and _results.xlsx resulting files saved in 
        the current working directory
    outputs: along with the _plots.xls and _results.xlsx in the working directory, the function returns 
    *  Results -  has the following important attributes and methods
        * .summary() - shows a summary of the regression results
        * .params - is a pandas series of the parameters fitted by the model
    """
    dates       = pd.Series(df.index)
    dependent   = df.iloc[:,0]
    independent = df.iloc[:,1:len(df.columns)]
    results = regress_data(dependent, independent)
    parameters = pd.Series(results.params)
    X = sm.add_constant(independent)
    fignum = 0
    # Generate test results for Breusch-Pagan LM test
    bptest = sms.het_breushpagan(results.resid, results.model.exog)
    
    plt.close()
    
    sm.qqplot(results.resid, fit = True, line = '45')
    plt.title('Q-Q Plot')
    plt.savefig('qq.jpg')
    plt.close()
    
    predicted = results.predict(X)
    plt.plot(dates, predicted, 'r--', dates, dependent, 'b--')
    plt.title('In Sample Backtesting')
    plt.xlabel('Date')
    plt.ylabel(dependent.name)
    plt.legend()
    plt.savefig('backtest.jpg')
    plt.close()    
    w = Workbook()
    ws = w.add_sheet('Plots')
    plot_to_excel('qq.jpg', w, ws, fignum)
    fignum +=1
    plot_to_excel('backtest.jpg', w, ws, fignum)
    fignum +=1
    
    for i in range(len(results.model.exog_names)):
        sm.graphics.plot_fit(results, results.model.exog_names[i])
        fig_name = results.model.exog_names[i] + '_fitted.jpg'
        plt.savefig(fig_name)
        plt.close()
        plot_to_excel(fig_name, w, ws, fignum)
        fignum +=1
    
    for i in range(len(independent.columns)):
        plt.scatter(independent.iloc[:,i], results.resid)
        plt.xlabel(independent.columns[i])
        plt.ylabel('Residuals')
        title_string = independent.columns[i] + ' vs. Residuals'
        plt.title(title_string)
        fig_name = independent.columns[i] + '_residuals.jpg'
        plt.savefig(fig_name)
        plt.close()
        plot_to_excel(fig_name, w, ws, fignum)
        fignum +=1
        
    save_plots   = out + '_plots.xls'
    save_results = out + '_results.xlsx'
    
    coef_plot(results, 'Coefficients.jpg')
    plot_to_excel('Coefficients.jpg', w, ws, fignum)
    fignum += 1
    
    writer = pd.ExcelWriter(save_results, engine='xlsxwriter')
    
    bptest = sms.het_breushpagan(results.resid, results.model.exog)
    bptest = pd.DataFrame([bptest[0], bptest[1]],['LM','P-Value'])
    bptest.to_excel(writer, sheet_name = "Breusch-Pagan")
    
    
    parameters         = pd.concat([results.params, results.bse, results.pvalues, results.conf_int()], axis=1)
    parameters.columns = ['Parameter', 'Std_Err', 'P-Value', 'Lower_Bound', 'Upper_Bound']
    parameters.to_excel(writer, sheet_name='Parameters')
    
    outliers = pd.concat([dates,results.outlier_test()], axis = 1)
    outliers = outliers.loc[outliers['bonf(p)'] <0.05 ]
    outliers.to_excel(writer, sheet_name = 'Outliers')
    
    rsquared_adj = pd.DataFrame([results.rsquared_adj])
    rsquared_adj.to_excel(writer, sheet_name = 'Adjusted R^2')
    
    aic = pd.DataFrame([results.aic])
    aic.to_excel(writer, sheet_name = 'AIC')
    
    bic = pd.DataFrame([results.bic])
    bic.to_excel(writer, sheet_name = 'BIC')
    
    dw = sm.stats.stattools.durbin_watson(results.resid)
    dw = pd.DataFrame([dw])
    dw.to_excel(writer, sheet_name = 'Durbin Watson')
        
    
    ftest = pd.DataFrame([results.fvalue, results.f_pvalue],['F','P-Value'])
    ftest.to_excel(writer, sheet_name = "F Test")
    
    mean_err = pd.DataFrame([results.mse_model, results.mse_resid, results.mse_total],['MSR','MSE', 'MSTO'])
    mean_err.to_excel(writer, sheet_name = "Mean Squared Error")
    
    cov_mat         =  pd.DataFrame(np.cov(np.transpose(independent)))
    cov_mat.columns = independent.columns
    cov_mat.index   = independent.columns
    cov_mat.to_excel(writer, sheet_name = 'Covariance Matrix')
    
    writer.save()
    w.save(save_plots)
    for file in os.listdir(os.getcwd()): 
        if (file.endswith('.bmp') or file.endswith('.jpg')):
            os.remove(file)
    return results
    
    
def regress_excel(filename, sheetname):
    """
    Define a function that takes in a data frame and a string to name the excel files produced and will returns results of 
    the regression performed
    inputs:
    * filename -  name of excel file containing the dates, dependent variable, and independent variables as columns, in that 
            order, will be used for both _plots.xls and _results.xlsx resulting files saved in the current working 
            directory
    * sheetname -  name of the sheet in the excel file desired, will be used for both _plots.xls and _results.xlsx resulting 
            files saved in the current working directory
    outputs: along with the [filename]_[sheetname]_plots.xls and _results.xlsx in the working directory, the function returns 
    *  Results -  has the following important attributes and methods
        * .summary() - shows a summary of the regression results
        * .params - is a pandas series of the parameters fitted by the model
    """
    dates, dependent, independent = extract_data(filename,sheetname)
    results = regress_data(dependent, independent)
    parameters = pd.Series(results.params)
    X = sm.add_constant(independent)
    fignum = 0
    # Generate test results for Breusch-Pagan LM test
    bptest = sms.het_breushpagan(results.resid, results.model.exog)
    
    plt.close()
    
    sm.qqplot(results.resid, fit = True, line = '45')
    plt.title('Q-Q Plot')
    plt.savefig('qq.jpg')
    plt.close()
    
    predicted = results.predict(X)
    plt.plot(dates, predicted, 'r--', dates, dependent, 'b--')
    plt.title('In Sample Backtesting')
    plt.xlabel('Date')
    plt.ylabel(dependent.name)
    plt.legend()
    plt.savefig('backtest.jpg')
    plt.close()    
    w = Workbook()
    ws = w.add_sheet('Plots')
    plot_to_excel('qq.jpg', w, ws, fignum)
    fignum +=1
    plot_to_excel('backtest.jpg', w, ws, fignum)
    fignum +=1
    
    for i in range(len(results.model.exog_names)):
        sm.graphics.plot_fit(results, results.model.exog_names[i])
        fig_name = results.model.exog_names[i] + '_fitted.jpg'
        plt.savefig(fig_name)
        plt.close()
        plot_to_excel(fig_name, w, ws, fignum)
        fignum +=1
    
    for i in range(len(independent.columns)):
        plt.scatter(independent.iloc[:,i], results.resid)
        plt.xlabel(independent.columns[i])
        plt.ylabel('Residuals')
        title_string = independent.columns[i] + ' vs. Residuals'
        plt.title(title_string)
        fig_name = independent.columns[i] + '_residuals.jpg'
        plt.savefig(fig_name)
        plt.close()
        plot_to_excel(fig_name, w, ws, fignum)
        fignum +=1
        
    save_plots   = filename[0:-5] + '_' + sheetname + '_plots.xls'
    save_results = filename[0:-5] + '_' + sheetname + '_results.xlsx'
    
    coef_plot(results, 'Coefficients.jpg')
    plt.close()
    plot_to_excel('Coefficients.jpg', w, ws, fignum)
    fignum += 1
    
    writer = pd.ExcelWriter(save_results, engine='xlsxwriter')
    
    bptest = sms.het_breushpagan(results.resid, results.model.exog)
    bptest = pd.DataFrame([bptest[0], bptest[1]],['LM','P-Value'])
    bptest.to_excel(writer, sheet_name = "Breusch-Pagan")
    
    
    parameters         = pd.concat([results.params, results.bse, results.pvalues, results.conf_int()], axis=1)
    parameters.columns = ['Parameter', 'Std_Err', 'P-Value', 'Lower_Bound', 'Upper_Bound']
    parameters.to_excel(writer, sheet_name='Parameters')
    
    outliers = pd.concat([dates,results.outlier_test()], axis = 1)
    outliers = outliers.loc[outliers['bonf(p)'] <0.05 ]
    outliers.to_excel(writer, sheet_name = 'Outliers')
    
    rsquared_adj = pd.DataFrame([results.rsquared_adj])
    rsquared_adj.to_excel(writer, sheet_name = 'Adjusted R^2')
    
    aic = pd.DataFrame([results.aic])
    aic.to_excel(writer, sheet_name = 'AIC')
    
    bic = pd.DataFrame([results.bic])
    bic.to_excel(writer, sheet_name = 'BIC')
    
    dw = sm.stats.stattools.durbin_watson(results.resid)
    dw = pd.DataFrame([dw])
    dw.to_excel(writer, sheet_name = 'Durbin Watson')
        
    
    ftest = pd.DataFrame([results.fvalue, results.f_pvalue],['F','P-Value'])
    ftest.to_excel(writer, sheet_name = "F Test")
    
    mean_err = pd.DataFrame([results.mse_model, results.mse_resid, results.mse_total],['MSR','MSE', 'MSTO'])
    mean_err.to_excel(writer, sheet_name = "Mean Squared Error")
    
    cov_mat         =  pd.DataFrame(np.cov(np.transpose(independent)))
    cov_mat.columns = independent.columns
    cov_mat.index   = independent.columns
    cov_mat.to_excel(writer, sheet_name = 'Covariance Matrix')
    
    writer.save()
    w.save(save_plots)
    for file in os.listdir(os.getcwd()): 
        if (file.endswith('.bmp') or file.endswith('.jpg')):
            os.remove(file)
    return results

def level_change(series, lag):
    """Compute the level change transformation on a series given a period over which to compute the change
    Inputs: 
    * Series - the data series to perform the transformation on 
    * Lag - the number of periods over which to perform the transformation
    Outputs: 
    * Returns the transformed data series
    """
    return series - series.shift(lag)

def log_change(series, lag):
    """
    Compute the log change transformation on a series given a period over which to compute the change
    Inputs: 
    * Series - the data series to perform the transformation on 
    * Lag - the number of periods over which to perform the transformation
    Outputs: 
    * Returns the transformed data series
    """
    return np.log(series / series.shift(lag))

def percent_change(series, lag):
    """
    Calculate the percent change transformation on a series given a period over which to compute the change
    Inputs: 
    * Series - the data series to perform the transformation on 
    * Lag - the number of periods over which to perform the transformation
    Outputs: 
    * Returns the transformed data series
    """
    return series / series.shift(lag) - 1

def build_derivatives(dataframe, max_lags = 4):
    """
    Construct the transformations on each series in a given data frame for periods 1-max_lags
    Inputs: 
    * dataframe - the dataframe to perform the transformation on 
    * max_lags - the maximum number of periods over which to perform the transformation
    Outputs: 
    * output -  transformed dataframe as a copy of the original plus the transformations
    """
    output = dataframe.copy()
    
    for column in output.columns:
        if not 'MEV' in column: continue
        for lag in range(1, max_lags + 1):
            output[column + '_LevelChange'   + str(lag)] = level_change(output[column], lag)
            output[column + '_LogChange'     + str(lag)] = log_change(output[column], lag)
            output[column + '_PercentChange' + str(lag)] = percent_change(output[column], lag)
            output[column + '_Lag'           + str(lag)] = output[column].shift(lag)
    output = output.replace(np.inf, np.nan)
    output = output.replace(-np.inf, np.nan)
    
    return output


def mev_preprocess(data, period = 'Q'):
    """
    Extract data from the MEV dataframe into a format where the index is a datetime column
    and each column of the dataframe is one of the variables. A dictionary is also returned with the MEV unique id, long name and short
    name. 
    This extraction relies on the use of 'Q_' in each of the quarterly date definitions in the file. 
    Inputs:
    * data - the MEV dataframe 
    Outputs:
    * mevs - the dataframe extracted from the file, indexed by date with the mevs in the columns
    * dictionary - a dataframe with columns based on the unique ID and the rows of unique id, short name, and long name
    """
    cols = []
    if 'q' in period.lower():
        for i in range(len(data.columns)):
            if 'Q_' in data.columns[i]:
                cols.append(i)
    else:
        for i in range(len(data.columns)):
            if 'M_' in data.columns[i]:
                cols.append(i)

    dates = data.columns[cols]
    mevs = data.loc[:,dates]
    mevs = mevs.transpose()
    mevs.columns = data['MEV_UNIQUE_ID_XL']
    dictionary = pd.DataFrame([data['MEV_UNIQUE_ID_XL'], data['MEV_LONG_NAME'], data['MEV_SHORT_NAME']])
    dictionary.columns = data['MEV_UNIQUE_ID_XL']
    date = []
    for i in range(len(mevs.index)):
        year = int(mevs.index[i][2:6])
        month = int(mevs.index[i][6:8])
        date.append(dt.datetime(year, month, 1))
    mevs['Date'] = date
    mevs.index = mevs['Date']
    mevs = mevs.drop(columns = ['Date'], axis = 1)
    return(mevs, dictionary)

def pull_scenario(year, test, scenario, folder, type = 'PNC', period = 'Q'):
    """
    # Pull the economics file from a given historical scenario and a given folder within the economics files saved 
    # locally from the import from MIP. Note this function does not import the file from MIP it relies on this having been done first.
    # Inputs:
    # * year - year of the stress test that the economic file was created for
    # * test - test that the economics file was created for 'CCAR' or 'DFAST'
    # * scenario - scenario of the forecast data desired: 'Base', 'Adverse', or 'Severe'
    # * folder - folder to pull the data from as can be found locally in the Economics folder
    # * type - defaults to 'PNC' but can be changed to 'Fed'
    """
    pwd = os.getcwd()
    if 'p' in str(type).lower():
        path = '/opt/app/pae/data/projects/pl59449/modeling/Economics/' + str(folder) + '/bhc/national/'
    else: 
        path = '/opt/app/pae/data/projects/pl59449/modeling/Economics/' + str(folder) + '/frb/national/'
    os.chdir(path)


    if 'c' in str(test).lower():
        file = 'd' + str(year-1) + '12'
    else: 
        file = 'd' + str(year) + '06'
    file = file + '_ent_'
    if 'p' in str(type).lower():
        file = file + 'p'
    else:
        file = file + 'f'

    if scenario[0].lower() == 'b':
        file = file + 'b'
    if scenario[0].lower() == 'a':
        file = file + 'a'
    if scenario[0].lower() == 's':
        file = file + 's'

    if 'q' in str(period).lower():
        file = file + '_qq_raw_'
    else: 
        file = file +  '_m_raw_'

    rate_file = file + 'rates_nat.sas7bdat'
    mev_file = file + 'mev_nat.sas7bdat'
    rates = pd.read_sas(rate_file, encoding='iso-8859-1')
    mevs  = pd.read_sas(mev_file, encoding='iso-8859-1')
    
    
    mevs, mev_dictionary = mev_preprocess(mevs, period = period)
    rates, rates_dictionary = mev_preprocess(rates, period = period)
    tot = pd.concat([rates, mevs], axis = 1)
    tot_dict = pd.concat([rates_dictionary, mev_dictionary], axis = 1, ignore_index = True)
    tot_dict.columns = tot_dict.loc['MEV_UNIQUE_ID_XL',:]
    os.chdir(pwd)
    return(tot, tot_dict)

def univariate_analysis(df, dep, dictionary):
    """
    Conduct univariate regressions on a dependent variable and ranks them based on adjusted r-squared
    Inputs:
    * df - dataframe with the independent variables to be used for the univariate regressions
    * dep - dataframe with the dependent variables that is the focus of the univariate analysis
    Outputs:
    * results.xlsx - excel file with variable, name, adjusted r-squared, and f-test p-value of univariate regression 
    """
    mat = pd.DataFrame(columns=['Name','Variable','Adj R-Squared','F Test P-Val'], index=df.columns)

    df = df.loc[dep.index]
    df = df.dropna(axis = 1)

    for i in range(len(df.columns)):
        results = regress_data(dep, df.iloc[:,i])
        var = df.columns[i]
        name = var
        if ('Change' or 'Lag') in var:
            name = var.rsplit('_',1)[0]

            name = dictionary.loc['MEV_LONG_NAME', name]
        
        mat.loc[df.columns[i]] = [name, var, results.rsquared_adj, results.f_pvalue]

            

    sortmat = mat.sort_values(by=['Adj R-Squared'], ascending = False)
    sortmat.to_excel('results.xlsx')
    return sortmat

def outlier_z_test(series, ztolerance = 3):
    """
    Test for outliers based on Z-score and returns a series with outliers removed and another series of the outliers
    Inputs:
    * series - series to test for outliers 
    * ztolerance - z-score tolerance for outlier determination, default is 3
    Outputs:
    * new_series - data series with outliers removed
    * outliers - data series containing hte outliers that were removed
    """
    z = np.abs(stats.zscore(series))
    outliers = series[z>ztolerance]
    new_series = series[z<ztolerance]
    return new_series, outliers


def find_mev(mevs, dictionary):
    """
    Select variables from mevs by allowing user to query the dictionary based on part of the name of the variable and asking clarifiying 
    questions to narrow down the search and ensure proper variable is seleected
    Inputs:
    * mevs - mevs data set from mev_pull_scenario function
    * dictionary - mev dictionary from mev_pull_scenario function
    * user-inputs - as the function runs, it will ask for user input to narrow the search
    Outputs:
    * mevs[selected] - the mev the user selects 
    """
    names = "dictionary.loc[&#39;MEV_UNIQUE_ID_XL&#39;]"
    long = "dictionary.loc[&#39;MEV_LONG_NAME&#39;,:]"
    Variable = input('&#39;Please" enter part of the variable name. ')
    count = "0"
    possibles = "[]"
    for i in range(len(long)):
        if Variable.upper() in long[i].upper():
            count += "1"
            print(count, long[i])
            possibles.append(names[i])
    if count == "0":
        print('Whoops! Try again with a different part of the variable name')
    elif count == "1":
        ans = input('&#39;Was" this the correct variable? ')
        if 'Y' in ans.upper():
            return mevs[possibles[0]]
        else:
            print('Whoops! Try again with a different part of the variable name')
    else:
        ans = input('&#39;Is" one of these your variable? ')
        if 'Y' in ans.upper():
            ans = input('&#39;Which" # option was the correct variable? ')
            return mevs[possibles[int(ans)]]
        else:
            print('Whoops! Try again with a different part of the variable name')
            
def check_tables(cnxn):
    """
    Display tables for an established SQL database connection. 
    Inputs:
    * cnxn - a pyodbc connection
    Outputs:
    * no return, just prints all table names 
    """
    cursor = "cnxn.cursor()"
    for row in cursor.tables():
        print(row.table_name)
        
def summStats(x):
    """
    Calculate the summary statistics for a variable x.
    Inputs:
    * x - variable to calculate summary statistics on
    Outputs:
    * Mean
    * Median
    * Minimum
    * Percentile1
    * Percentile5
    * percentile25
    * percentile75
    * percentile95
    * percentile99
    * maximum
    * Range
    * n - size of data
    * Standard deviation
    """
    x = "x[~np.isnan(x)]"
    mean = "np.mean(x)"
    median = "np.median(x)"
    minimum = "np.min(x)"

    percentile1 = "np.percentile(x,1)"
    percentile5 = "np.percentile(x,5)"
    percentile25 = "np.percentile(x,25)"

    percentile75 = "np.percentile(x,75)"
    percentile95 = "np.percentile(x,95)"
    percentile99 = "np.percentile(x,99)"
    maximum = "np.max(x)"

    ran = "maximum-minimum"
    n = "len(x)"

    std = "np.std(x)"
    
    return np.transpose([[mean, median, minimum, percentile1,percentile5,percentile25,percentile75,percentile95,percentile99,maximum, ran, n, std]])

            
            
#######################################
# ABC "interface" classes             #
#######################################

#######################################
# Abstract classes                    #
#######################################

#######################################
# Concrete classes                    #
#######################################

#######################################
# Initialization needed after member  #
#   definition is complete            #
#######################################

#######################################
# Imports needed after member         #
#   definition (to resolve circular   #
#   dependencies - avoid if at all    #
#   possible                          #
#######################################

#######################################
# Code to execute if the module is    #
#   called directly                   #
#######################################

if __name__ == '__main__':
    pass