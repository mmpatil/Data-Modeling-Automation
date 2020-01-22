import config as config
import ecm2 as ecm
import pandas as pd

df = ecm.wrangle_model_data(config.filename, config.sheetname, type='Log-Log')

df = df.loc[config.start:config.end]  # Filtering the dataframe based on the index of the rows

# print(df)

df = df.dropna(axis=1)

dep = df.iloc[:, 0]  # Taking all variables

# print("Dependent variable : ", dep)

dep.name = df.index[0]

# print('Dependent variable name :'+str(dep.name))

# print("Using index : ", df.index[1])
# print("Dep.name is ", dep.name)
# print(df)

# find the data generating process of the dependent variable
dgpa = ecm.find_dgp(dep, True)

# find the order of integration of the dependent variables
dep_order = ecm.integration_order(dep, alpha=config.adf_alpha)

base = ecm.exclusions_to_dummies(dep, config.exclusions)

# print("For Int Filter: \n",df)
regs = ecm.int_filter(df, dep_order, alpha=config.adf_alpha)
# print(regs)

candidates = ecm.statistical_testing(base, regs, adf_alpha=config.adf_alpha, param_alpha=config.param_alpha, bg_alpha=config.bg_alpha, white_alpha=config.white_alpha, sw_alpha=config.sw_alpha)

print(candidates)

mape = ecm.backtesting(candidates, base, regs, config.backtest_dates, config.backtest_long_dates)



