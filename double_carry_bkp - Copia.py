# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:29:19 2023

@author: ludmilla.tsurumaki
"""
from pandas.tseries.offsets import BDay
from datetime import date, timedelta
import matplotlib.pyplot as plt
import talib
import pandas as pd
import numpy as np
import vectorbt as vbt
from o3package import *

# import pdblp as bbg


df_ativo_final = pd.DataFrame()
df_pais_final = pd.DataFrame()
df_total_final = pd.DataFrame()
df_resumo_final = pd.DataFrame()
df_final = pd.DataFrame()

PAISES = ['BRL', 'MEX', 'CNH', 'AUD', 'EUR', 'ZAR']
# Lista de janelas (meses) para cálculo do momentum.
JANELAS = [16]
# Janela (meses) para vol de ajuste de posições.
JANELA_VOL = 4

today = date.today()

start = today - BDay(252*20)
start = date(2010, 1, 1)
end = date.today()-BDay(1)
start = start.strftime('%Y%m%d')
end = end.strftime('%Y%m%d')


px = pd.read_excel('series_term_all_data.xlsx', parse_dates=['date'])
px['date'] = px['date'].dt.strftime('%Y-%m-%d')
px.set_index('date', inplace=True)

px.sort_index(inplace=True)
px.fillna(method='ffill', inplace=True)

# Dicionários para tickers e ativos.

dic_tickers = pd.read_excel('tickers_double_carry.xlsx')
dic_tickers['ticker_tipo'] = dic_tickers['ticker'] + '_' + dic_tickers['tipo']
dic_tickers = dic_tickers[dic_tickers['pais'].isin(PAISES)]

juros = dic_tickers[dic_tickers['tipo'] == 'juros']
juros = dict(zip(juros.pais, juros.ticker))

bolsas = dic_tickers[dic_tickers['tipo'] == 'bolsa']
bolsas = dict(zip(bolsas.pais, bolsas.ticker))

moedas = dic_tickers[dic_tickers['tipo'] == 'moeda']
moedas = dict(zip(moedas.pais, moedas.ticker))

# Dicionário para ajuste das notações das moedas.

inverter = dic_tickers[dic_tickers['inv'] == 's']
inverter = list(inverter['ticker'])

data_complete = px[list(dic_tickers['ticker'])]
data_complete.dropna(inplace=True)


# ----------------------------------- MOMENTUM --------------------------------
# Cálculo do momentum dos juros em cada país e en cada janela.

for i in juros:
    for j in JANELAS:
        data_complete['mom_' + i + '_' +
                      str(j)] = data_complete[juros[i]] - data_complete[juros[i]].shift(21*j)


data_closes = pd.DataFrame()
for i in juros:
    for j in JANELAS:
        df = data_complete[[bolsas[i], moedas[i]]]
        if moedas[i] in inverter:
            # Ajuste das notações das moedas.
            df[moedas[i]] = 1/df[moedas[i]]
        else:
            pass
        df.rename(columns={bolsas[i]: bolsas[i]+str(j),
                  moedas[i]: moedas[i]+str(j)}, inplace=True)
        data_closes = pd.concat([data_closes, df], axis=1)

# Ajuste  do momentum para considerar a entrada no primeiro trade.

data_complete = data_complete.filter(regex='mom_')
data_ajuste = data_complete[data_complete.isnull().any(axis=1)]
data_ajuste = data_ajuste.index[-1]

data_complete = data_complete[data_ajuste:].fillna(0)

data_complete2 = data_complete.head(2)
data_complete2 = (data_complete2 - data_complete2.shift(-1)).dropna()

row_list = data_complete2.loc[data_ajuste, :].values.flatten().tolist()

data_complete.at[data_ajuste, data_complete.columns] = row_list

data_closes = data_closes[data_complete.dropna().index.values.min():]


# ---------------------------------- SINAIS -----------------------------------
# Criação dos sinais de entrada/saída.
# Sinais baseados na mudança de sinais do momentum.

l_entries_df = pd.DataFrame()
l_exits_df = pd.DataFrame()
s_entries_df = pd.DataFrame()
s_exits_df = pd.DataFrame()

for j in juros:
    for i in JANELAS:
        # Short:
        # Se o mom cross above 0: short_entry bolsa, short_entry moeda.
        s_entries = data_complete['mom_' + j +
                                  '_' + str(i)].vbt.crossed_above(0)

        # Se o mom cross below 0: short_exit bolsa, short_exit moeda.
        s_exits = data_complete['mom_' + j + '_' + str(i)].vbt.crossed_below(0)

        # Long:
        # Se o mom cross above 0: long_entry bolsa, long_entry moeda.
        l_entries = data_complete['mom_' + j +
                                  '_' + str(i)].vbt.crossed_below(0)

        # Se o mom cross below 0: long_exit bolsa, long_exit moeda
        # Note que tem que inverter a ordem das arrays.

        l_exits = data_complete['mom_' + j + '_' + str(i)].vbt.crossed_above(0)

        # Alinhando todos os DFs para os mesmos indices.

        entries_long_df = pd.DataFrame().reindex_like(
            data_closes[[bolsas[j] + str(i), moedas[j] + str(i)]])
        exits_long_df = pd.DataFrame().reindex_like(
            data_closes[[bolsas[j] + str(i), moedas[j] + str(i)]])

        entries_short_df = pd.DataFrame().reindex_like(
            data_closes[[bolsas[j] + str(i), moedas[j] + str(i)]])
        exits_short_df = pd.DataFrame().reindex_like(
            data_closes[[bolsas[j] + str(i), moedas[j] + str(i)]])

        # Longs
        entries_long_df[moedas[j] + str(i)] = l_entries
        entries_long_df[bolsas[j] + str(i)] = l_entries

        exits_long_df[moedas[j] + str(i)] = l_exits
        exits_long_df[bolsas[j] + str(i)] = l_exits

        # Shorts
        entries_short_df[moedas[j] + str(i)] = s_entries
        entries_short_df[bolsas[j] + str(i)] = s_entries

        exits_short_df[moedas[j] + str(i)] = s_exits
        exits_short_df[bolsas[j] + str(i)] = s_exits

        l_entries_df = pd.concat([l_entries_df, entries_long_df], axis=1)
        l_exits_df = pd.concat([l_exits_df, exits_long_df], axis=1)

        s_entries_df = pd.concat([s_entries_df, entries_short_df], axis=1)
        s_exits_df = pd.concat([s_exits_df, exits_short_df], axis=1)


data_closes = data_closes[l_entries_df.columns]


# --------------------------------- PORTFOLIO ---------------------------------
# Lista para facilitar identificação dos ativos.
lista_group = []
# Lista para ajustar colunas dos DFs de entradas/saídas.
lista_pos = []

for j in juros:
    for i in JANELAS:
        lista_pos.append(bolsas[j] + str(i))
        lista_group.append('bolsa_' + j)
    for i in JANELAS:
        lista_pos.append(moedas[j] + str(i))
        lista_group.append('moeda_' + j)

# Colocando DFs com colunas nas mesmas ordens.

data_closes = data_closes[lista_pos]
l_entries_df = l_entries_df[lista_pos]
l_exits_df = l_exits_df[lista_pos]
s_entries_df = s_entries_df[lista_pos]
s_exits_df = s_exits_df[lista_pos]

pf3 = vbt.Portfolio.from_signals(data_closes,
                                 entries=l_entries_df,
                                 exits=l_exits_df,
                                 short_entries=s_entries_df,
                                 short_exits=s_exits_df,
                                 init_cash=1000,
                                 cash_sharing=False,
                                 group_by=lista_group,
                                 freq='d',
                                 log=True)

# ----------------------------------- PESOS -----------------------------------

serie_ret = pf3.returns()
serie_ret = serie_ret[~(serie_ret == 0).all(axis=1)]
# Ajustes dos nomes _bolsa e _moeda.
dic_tickers['value'] = dic_tickers['tipo'] + '_' + dic_tickers['pais']
dic_tickers2 = dic_tickers[dic_tickers['tipo'].isin(['bolsa', 'moeda'])]
dic_tickers2 = dict(zip(dic_tickers2.ticker, dic_tickers2.value))

px2 = px[dic_tickers2]
px2 = px2.rename(columns=dic_tickers2)

# Pesos = inverse vol.

vol = px2.pct_change().dropna().rolling(
    21 * JANELA_VOL).std() * np.sqrt(252) * 100

inv_vol = 1/vol
w_bolsa_moeda = inv_vol.shift(1).dropna()
w_bolsa_moeda = w_bolsa_moeda.reindex_like(serie_ret).dropna()

# Retornos ponderados e acumulados (bolsa e moeda de cada país).
port = w_bolsa_moeda * serie_ret
port_ativo_pais = port.add(1).cumprod() - 1

# Retornos ponderados e acumulados (agregado de cada país).
port_pais = port.copy()
for i in PAISES:
    port_pais[i] = port_pais[[
        col for col in port_pais.columns if col.endswith('_' + i)]].sum(axis=1)

port_pais = port_pais[PAISES]
port_pais_acum = port_pais.add(1).cumprod() - 1


# Retornos ponderados e acumulados (agregado total).
# Equal weighted por país (1/#paises).

vol_pais = port_pais.rolling(21*JANELA_VOL).std().dropna()

w_pais = pd.DataFrame().reindex_like(vol_pais)
w_pais[w_pais.columns] = 1/len(w_pais.columns)

port_total = port_pais.copy()
port_total = port_total * w_pais
port_total = pd.DataFrame(port_total.sum(axis=1), columns=['port_total'])

port_total_acum = port_total.add(1).cumprod() - 1


# Métricas de risco.
vol1 = port.std() * np.sqrt(252)
ret1 = port.apply(lambda x: np.prod(1 + x) - 1).add(1).pow(252/len(port)) - 1
sharpe1 = ret1/vol1

vol2 = port_pais.std() * np.sqrt(252)
ret2 = port_pais.apply(lambda x: np.prod(
    1 + x) - 1).add(1).pow(252/len(port_pais)) - 1
sharpe2 = ret2/vol2

vol3 = port_total.std() * np.sqrt(252)
ret3 = port_total.apply(lambda x: np.prod(
    1 + x) - 1).add(1).pow(252/len(port_total)) - 1
sharpe3 = ret3/vol3

vol_final = vol1.append([vol2, vol3])


# Ajuste DF
vol_final = vol1.append([vol2, vol3])
ret_final = ret1.append([ret2, ret3])
sharpe_final = sharpe1.append([sharpe2, sharpe3])

resumo = [vol_final, ret_final, sharpe_final]

resumo = pd.DataFrame(resumo).T
resumo.rename(columns={0: 'vol', 1: 'ret', 2: 'sharpe'}, inplace=True)
resumo.reset_index(inplace=True)
resumo['janela'] = JANELAS[0]


port_ativo_pais['tipo'] = 'ret_acum'
w_bolsa_moeda['tipo'] = 'weight'
port_ativo_pais = pd.concat([port_ativo_pais, w_bolsa_moeda])

vol_pais['tipo'] = 'vol'
port_pais_acum['tipo'] = 'ret_acum'
port_pais_acum = pd.concat([vol_pais, port_pais_acum])

# Criação do DF com as posicões na ponta.

ajuste_tickers = dict(zip(dic_tickers.value, dic_tickers.ticker))
positions = w_bolsa_moeda.drop(columns='tipo').tail(1).T
positions.reset_index(inplace=True)
positions['vol'] = vol3[0]
positions['sharpe'] = sharpe3[0]

positions.columns = ['prod', 'weight', 'vol', 'sharpe']
positions['ticker'] = positions['prod'].map(ajuste_tickers)
positions = positions[['prod', 'ticker', 'weight', 'vol', 'sharpe']]
positions['weight'] = positions['weight']/len(w_pais.columns)

pos_vbt = pf3.positions
pos_vbt = pos_vbt.records_readable
pos_vbt = pos_vbt[pos_vbt['Status'] == 'Open']
pos_vbt = pos_vbt[['Column', 'Direction']]
pos_vbt['Column'] = pos_vbt['Column'].str.replace(str(JANELAS[0]), '')
pos_vbt.columns = ['ticker', 'direction']

positions_final = positions.merge(pos_vbt)
positions_final['weight'] = np.where(
    positions_final['direction'] == 'Short', -positions_final['weight'], positions_final['weight'])

positions_final.drop(columns='direction', inplace=True)

resumo['date'] = today
positions_final['date'] = today

resumo = resumo[['date', 'index', 'vol', 'ret', 'sharpe', 'janela']]
positions_final = positions_final[[
    'date', 'prod', 'ticker', 'weight', 'vol', 'sharpe']]
'''
positions_final.to_excel('pos_teste2.xlsx')
resumo.to_excel('res_teste2.xlsx')
'''
