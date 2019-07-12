"""
用来分析各个策略之间的相关性
"""

import pandas as pd
import itertools as it
import os
import matplotlib.pyplot as plt

strategy_list = ['current', 'AnnualRtn', 'sharp', 'MaxDrawdown']
file_path = 'F:\\BackTestSystem3\\strategy\\profit_rate'

# 根据各个策略的净值计算其相关性
df_rtn = pd.DataFrame()
for s in strategy_list:
    df = pd.read_csv(os.path.join(file_path, 'result_'+s+'.csv'), index_col=0, parse_dates=True)[['净值']]
    df.rename(columns={'净值': s}, inplace=True)
    df_rtn = df_rtn.join(df, how='outer')
    df_rtn = df_rtn.diff(periods=1)
total_corr = df_rtn.corr()
print('============总的相关性===========')
print(total_corr)

# 计算滚动相关系数
df_rolling_corr = pd.DataFrame()
for comb in it.combinations(df_rtn.columns, 2):
    df1 = df_rtn[comb[0]]
    df2 = df_rtn[comb[1]]
    df_corr = df1.rolling(window=20).corr(df2).to_frame(name='%s&%s' % (comb[0], comb[1]))
    df_rolling_corr = df_rolling_corr.join(df_corr, how='outer')
df_rolling_corr.plot(label=True, grid=True)
plt.show()
