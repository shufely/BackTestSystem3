#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@time  : 2019/5/27 11:18
@author: Li Ye
"""

from lib.simulator.base import BacktestSys, HoldingClass
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import traceback


class Basic(BacktestSys):

    def get_fundamental_data(self):
        future_df, spot_df, inventory_df, profit_df = pd.DataFrame(index=self.dt), pd.DataFrame(index=self.dt), \
                                                      pd.DataFrame(index=self.dt), pd.DataFrame(index=self.dt)
        for v in self.data['spot_price'].values():
            if v.commodity == 'FU':
                spot_df[v.commodity] = v.CLOSE * self.dollar2rmb.CLOSE
            elif 'CLOSE' in v.__dict__:
                spot_df[v.commodity] = v.CLOSE
            elif 'price' in v.__dict__:
                spot_df[v.commodity] = v.price

        for v in self.data['inventory'].values():
            if v.commodity == 'J':
                if v.commodity in inventory_df:
                    inventory_df[v.commodity] += v.CLOSE
                else:
                    inventory_df[v.commodity] = v.CLOSE
            elif 'CLOSE' in v.__dict__:
                inventory_df[v.commodity] = v.CLOSE
            elif 'inventory' in v.__dict__:
                inventory_df[v.commodity] = v.inventory

        for v in self.data['profit_rate'].values():
            profit_df[v.commodity] = v.upper_profit_rate

        for k, v in self.data['future_price'].items():
            future_df[v.commodity] = getattr(v, self.conf['backtest_mode'])
            if v.commodity not in spot_df:
                raise LookupError('{}的现货价格数据未提取！'.format(v.commodity))
            if v.commodity not in inventory_df:
                raise LookupError('{}的库存数据未提取！'.format(v.commodity))
            if v.commodity not in profit_df:
                raise LookupError('{}的加工利润率数据未提取！'.format(v.commodity))

        return {'future_df': future_df, 'spot_df': spot_df, 'inventory_df': inventory_df, 'profit_df': profit_df}

    def var_forecast(self, steps, **arrays,):
        arrays = pd.DataFrame.from_dict(arrays).dropna().to_dict(orient='series')

        # 对自变量做标准化处理
        for i, array in enumerate(arrays.values()):
            if i == 0:
                continue
            else:
                key = list(arrays.keys())[i]
                arrays[key] = (array-array.mean())/array.std()

        vectors = np.stack(list(arrays.values()), axis=1)  # axis=1表示以第二维度（列维度）为方向进行向量堆叠
        y = vectors[:, 0]  # 因变量
        x = vectors[:, 1:]  # 自变量

        # 序列ADF平稳性检验
        for name, array in arrays.items():
            adfuller_pvalue = sm.tsa.stattools.adfuller(array, regression='c', autolag='AIC')[1]
            if adfuller_pvalue > 0.05:
                print('  {}序列在5%的置信度下不平稳'.format(name))
                return

        # 因变量与自变量的协整检验
        coint_pvalue = sm.tsa.stattools.coint(y, x, trend='c', method='aeg', maxlag=None, autolag='aic')[1]
        if coint_pvalue > 0.05:
            print('  在5%的置信度下因变量与各自变量不存在长期协整关系')
            return

        # 模型估计&定阶
        varMod = sm.tsa.VAR(vectors)
        fitMod = varMod.fit(method='ols', ic='aic', trend='c', verbose=False)  # verbose=True显示阶数选择结果
        if fitMod.k_ar > 0:
            forecast = fitMod.forecast(vectors, steps=steps)
            forecast_dict = dict(zip(arrays.keys(), forecast[0]))
        else:
            print('  VAR模型的滞后阶数选择为0，无法进行forecast估计')
            return

        # 判断VAR模型的稳定性 AR根图
        resid_unitroot_pvalue = sm.stats.diagnostic.breaks_cusumolsresid(fitMod.resid, ddof=x.shape[1])[1]
        if resid_unitroot_pvalue < 0.05:
            print('  VAR模型存在单位根')
            return

        return forecast_dict

    def strategy_1(self):
        holdings = HoldingClass(self.dt)

        data_df = self.get_fundamental_data()

        for k, v in self.data['future_price'].items():
            print('\r' + v.commodity)
            holdings.add_holdings(k, np.zeros(len(self.dt)))

            vectors_df = pd.DataFrame()
            vectors_df['return'] = data_df['future_df'][v.commodity].pct_change(periods=1, fill_method='ffill')
            vectors_df['basis_diff'] = (data_df['spot_df'][v.commodity]-data_df['future_df'][v.commodity]).ffill().diff()
            vectors_df['inventory_pct_diff'] = data_df['inventory_df'][v.commodity].pct_change(periods=5, fill_method='ffill').diff()
            vectors_df['profit_diff'] = data_df['profit_df'][v.commodity].ffill().diff()

            temp_df = vectors_df.copy()
            temp_df.set_index(self.dt, inplace=True)
            temp_df.to_csv(r'C:\Users\uuuu\Desktop\\'+k+'.csv', encoding='gbk')

            observation_num = vectors_df.count().min()
            start = len(self.dt) - observation_num

            for i in range(start+120, len(self.dt)):    # 最少有120个样本
                print('\r'+self.dt[i].strftime('%Y-%m-%d'), end='')
                arrays = vectors_df.iloc[start:i+1, :].to_dict(orient='series')
                prediction = self.var_forecast(steps=1, **arrays)  # 传入**kwarg参数，用字典形式时需要加**，用赋值形式时不需要**
                if prediction:
                    if prediction['return'] >= 0:
                        getattr(holdings, k)[i] = 1
                    else:
                        getattr(holdings, k)[i] = -1

        return holdings

    def strategy_2(self, on):
        holdings_num = 3
        holdings = HoldingClass(self.dt)

        bs_df = 1-future_df/spot_df
        iv_df = inventory_df.pct_change(periods=5)+1
        if on == 'iv':
            key_df = iv_df
            # key_df = inventory_df/inventory_df.rolling(window=30, min_periods=1).mean().shift(periods=235)
            rank_df = key_df.rank(ascending=0, method='first', axis=1)
        elif on == 'basis':
            key_df = bs_df
            rank_df = key_df.rank(ascending=1, method='first', axis=1)
        elif on == 'basis_iv':
            bs_rank = bs_df.rank(axis=1)
            iv_rank = iv_df.rank(axis=1)
            # key_df = bs_rank / iv_df
            # key_df = bs_rank / iv_rank
            # key_df = bs_rank - iv_df
            rank_df = key_df.rank(ascending=1, method='first', axis=1)

        rank_df['count'] = rank_df.count(axis=1)

        for k, v in self.data['future_price'].items():
            holdings.add_holdings(k, np.zeros(len(self.dt)))
            con_short = rank_df[v.commodity] <= rank_df['count'].map(lambda x: min(x/2, holdings_num))
            con_long = rank_df[v.commodity] > rank_df['count'].map(lambda x: max(math.ceil(x/2), x-holdings_num))

            getattr(holdings, k)[con_short] = -1
            getattr(holdings, k)[con_long] = 1

        return holdings


if __name__ == '__main__':
    ins = Basic()
    # data = ins.get_fundamental_data()

    holdings = ins.strategy_1()
    # holdings = ins.strategy_2(on='basis_iv')
    holdings.to_frame().to_csv(r'C:\Users\uuuu\Desktop\holding_df1.csv', encoding='gbk')
    ins.holdingsStandardization(holdings, mode=2)
    holdings = ins.holdingsProcess(holdings)
    holdings.to_frame().to_csv(r'C:\Users\uuuu\Desktop\holding_df2.csv', encoding='gbk')

    ins.displayResult(holdings)




