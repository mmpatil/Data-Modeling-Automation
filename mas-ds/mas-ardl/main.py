#!/usr/bin/python

import config
import ecm as ecm
import datetime as dt
import pandas as pd

df = ecm.wrangle_model_data(config.filename, config.sheetname, type='Log-Log')

df = df.loc[config.start:config.end]
df = df.dropna(axis=1)

dep = df.iloc[:, 0]
dep.name = df.index[0]

# find the data generating process of the dependent variable
dgpa = ecm.find_dgp(dep)

# find the order of integration of dependent variable
dep_order =  ecm.integration_order(dep, alpha=config.adf_alpha)

base = ecm.exclusions_to_dummies(dep, config.exclusions)
regs = ecm.int_filter(df, dep_order, alpha=config.adf_alpha)

candidates = ecm.statistical_testing(base, regs, adf_alpha=config.adf_alpha, param_alpha=config.param_alpha,
                                     bg_alpha=config.bg_alpha, white_alpha=config.whit_alpha, sw_alpha=config.sw_alpha)

print("Candidates :", candidates)

mape = ecm.backtesting(candidates, base, regs, config.backtest_dates, config.backtest_long_dates)

# print("MAPE : ", mape)

try:
    short_list = config.short_list
except:
    short_list = candidates

for i in short_list:
    X = ecm.create_design(base, regs, i)
    print("X : ", X)
    print("Dep : ", dep)
    print("Dep Type : ", type(dep))
    print("X.index : ", X.index)
    newdep = dep[X.index]
    n = X.index.get_loc(config.rq)
    params, ps = ecm.recursive_reg(newdep, X, n, varname=i)

ecm.stress_test_plot(config.filename, config.shtm, config.shtb, config.shta, config.shts, short_list, config.pq0,
                     config.pq1, base, regs, config.bottom, config.top)

new_model_summary, old_model_summary = ecm.stress_test_compare(config.filename, config.shtm, config.shtb, config.shta,
                                                               config.shts, config.shtc, short_list, config.pq0,
                                                               config.pq1, base, regs, dep, bottom=config.bottom,
                                                               top=config.top)

ecm.create_sensitivity(config.filename, config.shtm, config.shtb, base, regs, short_list, config.pq0, config.pq1)

pqs = ['2008-03-31', '2012-03-31', '2015-03-31', '2016-03-31']
ecm.out_of_time(candidates, base, regs, pd.to_datetime('2014-12-31'), pqs)

ecm.compile_results(short_list)

ecm.copy_output('key_results')

folder = str(dt.datetime.now())[:16]
ecm.version_output(folder)
