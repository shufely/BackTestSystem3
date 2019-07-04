#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@time  : 2019/3/4 12:43
@author: Li Ye
"""

from lib.simulator.base import BacktestSys, HoldingClass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy
import yaml
import os
import re
import sys

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Reversion(BacktestSys):
    def __init__(self):
        super().__init__()
        self.abg_dict = self.conf['arbitrage_library']
        self.ctr_dict = {}
        for k, v in self.data['future_price'].items():
            self.ctr_dict[v.commodity] = k

        # 可调参数
        self.period_1 = 30  # 定义计算回归均值时的基准区间长度
        self.period_2 = 90  # 定义计算偏离的均值标准差时的基准区间长度
        self.return_pt = 2  # 定义止盈点位/收益区间长度
        self.risk_pt = 1  # 定义止损点位/风险区间长度, 如果要改变风险收益区间，下面的条件判断逻辑也要改变

    def spread(self, cmd_list, formula):
        df = pd.DataFrame(index=self.dt)
        formula = str(sympy.expand(formula))  # 展开表达式，将和的乘积全部展开

        for i in range(len(cmd_list)):
            obj_content = self.ctr_dict[cmd_list[i]]
            df[cmd_list[i]] = self.data['future_price'][obj_content].CLOSE
            # TODO: 这里应该用backtest_mode定义使用CLOSE还是OPEN吗？
            formula = formula.replace('var'+str(i+1), 'df[{!r}]'.format(cmd_list[i]))  # {!r}作用是在填充前先用repr()函数来处理参数,即加引号
        if 'ExRate' in formula:
            df['ExRate'] = self.dollar2rmb.CLOSE
            formula = formula.replace('ExRate', "df['ExRate']")
        df['spread'] = eval(formula)

        formula_split = formula.replace('-', '+').split('+')
        for part in formula_split:
            if 'df' not in part:
                formula_split.remove(part)
        formula_short = '+'.join(formula_split)
        df['capital'] = eval(formula_short)

        return df

    def seasonal_bias(self, cmd_list, formula):
        df = self.spread(cmd_list, formula)
        df['year'] = df.index.year.map(str)
        df['mm-dd'] = df.index.strftime('%m-%d')
        seasonal_df = df.pivot(index='mm-dd', columns='year', values='spread')
        seasonal_df.dropna(axis=1, how='all', inplace=True)  # 去掉全部是NaN值的年份，因为总的时间区间是self.dt
        seasonal_df.fillna(method='ffill', inplace=True)
        seasonal_df.iloc[:, 1:] = seasonal_df.iloc[:, 1:].bfill()
        seasonal_df[str(max(map(int, seasonal_df.columns))+1)] = np.nan
        seasonal_mean = seasonal_df.rolling(window=3, min_periods=3, axis=1).mean().shift(axis=1)

        mean_df = seasonal_mean.T.stack().to_frame('seasonal_mean')
        # stack()将列索引转成双层索引的内层行索引,to_frame将series转为dataframe
        mean_df.index = pd.to_datetime(['-'.join(x) for x in mean_df.index.ravel()], errors='coerce')
        # MultiIndex.ravel()将双层索引降维，to_datetime的errors='coerce'表示invalid parsing will be set as NaT，用于处理平年的02-29
        mean_df = mean_df.rolling(window=self.period_1).mean().shift(periods=-self.period_1)  # 对季节性均值做30日移动平均
        df = df[['spread', 'capital']].join(mean_df)

        df['diff'] = (df['seasonal_mean']-df['spread'])/df['capital']
        df['diff_mean'] = df['diff'].rolling(window=self.period_2, min_periods=self.period_2-10).mean()
        df['diff_std'] = df['diff'].rolling(window=self.period_2, min_periods=self.period_2-10).std()
        df['seasonal_bias'] = (df['diff']-df['diff_mean'])/df['diff_std']

        return df['seasonal_bias'].values

    def trend_bias(self, cmd_list, formula):
        df = self.spread(cmd_list, formula)
        df['trend_mean'] = df['spread'].rolling(window=self.period_1).mean()
        df['diff'] = (df['trend_mean']-df['spread'])/df['capital']
        df['diff_mean'] = df['diff'].rolling(window=self.period_2, min_periods=self.period_2-10).mean()
        df['diff_std'] = df['diff'].rolling(window=self.period_2, min_periods=self.period_2-10).std()
        df['trend_bias'] = (df['diff']-df['diff_mean'])/df['diff_std']

        return df['trend_bias'].values

    def strategy(self, open_pt=3, indicator='S', abg=None):
        # indicator可选S-seasonal_bias/T-trend_bias/WA-weighted_average_bias
        # abg可选套利组合名称，返回单个套利组合的仓位；abg默认None，返回所有套利组合加总后的整体头寸
        if (abg is not None) and (abg not in self.abg_dict.keys()):
            raise KeyError('套利组合名称无效')

        long_sp = open_pt - self.return_pt
        long_sl = open_pt + self.risk_pt
        short_sp = -open_pt + self.return_pt
        short_sl = -open_pt - self.risk_pt

        holdings = HoldingClass(self.dt)
        abg_holdings = {}

        for k, v in self.abg_dict.items():
            cmd_list = k.split('-')
            pattern = re.compile('var\d+')
            var_list = np.unique(pattern.findall(v['formula']))  # findall返回匹配到的所有非重复字串的列表, 好像并不能去重？？
            if len(cmd_list) != len(var_list):
                raise ValueError('公式中的变量个数异常')

            ctr_list = [self.ctr_dict[c] for c in cmd_list]

            if indicator == 'S':
                bias = self.seasonal_bias(cmd_list, v['formula'])
            elif indicator == 'T':
                bias = self.trend_bias(cmd_list, v['formula'])
            elif indicator == 'WA':
                bias = self.seasonal_bias(cmd_list, v['formula'])/2 + self.trend_bias(cmd_list, v['formula'])/2
            else:
                raise ValueError
            if len(bias) != len(self.dt):
                raise Exception('偏离指标长度与交易时间长度不一致')

            for cmd in ctr_list:
                if cmd not in holdings.__dict__:
                    holdings.add_holdings(cmd, np.zeros(len(self.dt)))
            abg_holdings[k] = HoldingClass(self.dt)

            position = np.zeros(len(self.dt))

            for i in range(1, len(self.dt)):
                try:
                    direction = v['direction'][str(self.dt[i].year)]
                except KeyError:
                    direction = 0

                if bias[i-1] >= long_sl:
                    position[i] = 0
                elif bias[i-1] >= open_pt:
                    if position[i-1] > 0:
                        position[i] = position[i-1]
                    elif direction != -1:
                        position[i] = 1
                    else:
                        position[i] = 0
                elif bias[i-1] > max(long_sp, short_sp):
                    if position[i-1] > 0:
                        position[i] = position[i-1]
                    else:
                        position[i] = 0
                elif bias[i-1] >= min(long_sp, short_sp):
                    if long_sp >= short_sp:
                        position[i] = 0
                    else:
                        position[i] = position[i - 1]
                elif bias[i-1] > -open_pt:
                    if position[i-1] < 0:
                        position[i] = position[i-1]
                    else:
                        position[i] = 0
                elif bias[i-1] > short_sl:
                    if position[i-1] < 0:
                        position[i] = position[i-1]
                    elif direction != 1:
                        position[i] = -1
                    else:
                        position[i] = 0
                else:
                    position[i] = 0

            for j in range(len(ctr_list)):
                cof = sympy.Poly(sympy.expand(v['formula'])).as_expr().coeff('var'+str(j+1))
                if 'ExRate' in str(cof):
                    cof = sympy.Poly(str(cof)).coeff_monomial('ExRate')
                    # Poly.as_expr().coeff(x)与Poly.coeff_monomial(x)都是返回系数，但有差异

                holdings.update_holdings(ctr_list[j], getattr(holdings, ctr_list[j]) + (position * cof))
                abg_holdings[k].add_holdings(ctr_list[j], position * cof)

        self.holdingsStandardization(holdings, mode=0)
        holdings = self.holdingsProcess(holdings)

        if abg is None:
            return holdings
        elif abg in self.abg_dict.keys():
            return abg_holdings[abg]

    def optimal_open(self, criterion='NV', indicator='S', abg=None):
        # criterion可选 期末净值-NV, 年化收益率-Yields, 夏普比率-Sharpe

        nv, yields, sharpe, drawback, trade_times = [[]]*5
        # print(nv, yields, sharpe, drawback, trade_times)

        for open_pt in np.arange(1, 4, 0.1):
            holdings = self.strategy(open_pt, indicator, abg)

            nv.append(self.getNV(holdings)[-1])
            # yields.append(self.calcIndicator(holdings)[0])
            # sharpe.append(self.calcIndicator(holdings)[2])
            # drawback.append(self.calcIndicator(holdings)[3])

            self.ProgressBar((open_pt - 1) / 0.1 + 1, len(np.arange(1, 4, 0.1)))

        if criterion == 'NV':
            open_pt_optimal = nv.index(max(nv)) * 0.1 + 1
        elif criterion == 'Yields':
            open_pt_optimal = yields.index(max(yields)) * 0.1 + 1
        elif criterion == 'Sharpe':
            open_pt_optimal = sharpe.index(max(sharpe)) * 0.1 + 1
        else:
            raise KeyError

        print(nv, yields, sharpe, drawback)
        print(open_pt_optimal)

        plt.figure()
        plt.subplot(311)
        plt.plot(np.arange(1, 4, 0.1), nv, label='期末净值')
        # plt.xlabel('建仓点')
        # plt.ylabel('期末净值')
        plt.legend()

        plt.subplot(312)
        plt.plot(np.arange(1, 4, 0.1), drawback, label='最大回撤')
        plt.legend()

        plt.subplot(313)
        plt.plot(np.arange(1, 4, 0.1), sharpe, label='夏普比率')
        plt.legend()

        plt.show()

        return open_pt_optimal

    def ProgressBar(self, current_steps, total_steps, progress_width=50):
        width_done = int((current_steps / total_steps) * progress_width)
        width_rest = progress_width - width_done
        progress_percent = current_steps / total_steps

        progress_bar = '|' + '>' * width_done + '-' * width_rest + '|'

        if current_steps == total_steps:
            progress_bar += '\n'

        sys.stdout.write('\r' + 'Progress:{:.1%}'.format(progress_percent) + progress_bar)
        sys.stdout.flush()


if __name__ == '__main__':
    ins = Reversion()
    holdings = ins.strategy(open_pt=1.4, indicator='S')
    # open_pt_optimal = ins.optimal_open(criterion='NV', indicator='S')
    ins.displayResult(holdings)



