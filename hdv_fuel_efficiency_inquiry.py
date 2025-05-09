# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:06:51 2025

@author: g_s_s
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# import seaborn as sns
# import random
import streamlit as st
import ast
import pickle


pd.options.mode.copy_on_write = True
fm.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = 'Taipei Sans TC Beta'

st.set_page_config(
    page_title="功能打樣版 僅供3人同時使用",
    page_icon="random",
)


def model_1(): # 能效=f(車速)
    st.write("可自行上傳逐秒之車速、車重資料檔案(.csv)，需以「,」區隔欄位(columns)。(註: 車重=空車重+載重)")
    st.write("檔案的第一列(row)需有「VehicleWeight[ton], VehicleSpeed[km/h]」兩欄位名稱, 其餘列為數字資料。")  
    st.write("")
    uploaded_file=st.file_uploader("選擇上傳檔案：",type=".csv")    
    st.write("")
    if uploaded_file:
        df=pd.read_csv(uploaded_file, usecols=['VehicleWeight[ton]', 'VehicleSpeed[km/h]'])
        st.write("已上傳檔案：", uploaded_file.name)
        st.write(f"上傳的行駛操作資料(共{len(df)}筆紀錄)：")
    else:
        df=pd.read_csv('default_data.csv')
        st.write(f"預設的行駛操作資料(共{len(df)}筆紀錄)：")   

    st.dataframe(df)

    pp_FULL = np.load("model_1_spacing_5km_FULL_LOAD.npy",allow_pickle=True) # 14.975+(26-14.975)*0.9
    pp_HALF = np.load("model_1_spacing_5km_HALF_LOAD.npy",allow_pickle=True) # 14.975+(26-14.975)*0.55
    
    df['time[s]'] = [i for i in range(1, len(df)+1)]
    VW_FULL_LOAD = (14.975+(26-14.975)*0.9)
    VW_HALF_LOAD = (14.975+(26-14.975)*0.55)

    df['predict_FuelRate[L/h]'] = df.apply(lambda x: np.poly1d(pp_HALF)(x['VehicleSpeed[km/h]']) +
                                      (np.poly1d(pp_FULL)(x['VehicleSpeed[km/h]'])-np.poly1d(pp_HALF)(x['VehicleSpeed[km/h]'])) /
                                      (VW_FULL_LOAD-VW_HALF_LOAD) *
                                      (x['VehicleWeight[ton]']-VW_HALF_LOAD), axis=1)    
    
    st.subheader(f"累計行駛 {df['VehicleSpeed[km/h]'].sum()/3.6/1000:.2f} 公里")
    st.subheader(f"預測用油 {df['predict_FuelRate[L/h]'].sum()/3600:.2f} 公升")
    st.subheader(f"預測能效 {(df['VehicleSpeed[km/h]'].sum()/3.6/1000)/(df['predict_FuelRate[L/h]'].sum()/3600):.2f} 公里/公升")


def model_5():
    st.write("可自行上傳逐秒之車速、引擎轉速、車重資料檔案(.csv)，需以「,」區隔欄位(columns)。(註: 車重=空車重+載重)")
    st.write("檔案的第一列(row)需有「VehicleWeight[ton], VehicleSpeed[km/h], EngineSpeed[rpm]」三欄位名稱, 其餘列為數字資料。")
    st.write("")
    uploaded_file=st.file_uploader("選擇上傳檔案：",type=".csv")    
    st.write("")
    if uploaded_file:
        df=pd.read_csv(uploaded_file, usecols=['VehicleWeight[ton]', 'VehicleSpeed[km/h]', 'EngineSpeed[rpm]'])
        st.write("已上傳檔案：", uploaded_file.name)
        st.write(f"上傳的行駛操作資料(共{len(df)}筆紀錄)：")
    else:
        df=pd.read_csv('default_data.csv')
        st.write(f"預設的行駛操作資料(共{len(df)}筆紀錄)：")  

    st.dataframe(df)

    # 14.975+(26-14.975)*0.9
    with open('model_5_spacing_6x6_FULL_LOAD.pkl', 'rb') as f:
        reg_FULL_LOAD = pickle.load(f)
    # 14.975+(26-14.975)*0.55
    with open('model_5_spacing_6x6_HALF_LOAD.pkl', 'rb') as f:
        reg_HALF_LOAD = pickle.load(f)

    VW_FULL_LOAD = 14.975+(26-14.975)*0.9 # [ton]
    VW_HALF_LOAD = 14.975+(26-14.975)*0.55 # [ton]
    df['predict_FULL_LOAD_FuelRate[L/h]'] = reg_FULL_LOAD.predict(df[['VehicleSpeed[km/h]', 'EngineSpeed[rpm]']])
    df.loc[df['predict_FULL_LOAD_FuelRate[L/h]']<0, 'predict_FULL_LOAD_FuelRate[L/h]'] = 0.
    df['predict_HALF_LOAD_FuelRate[L/h]'] = reg_HALF_LOAD.predict(df[['VehicleSpeed[km/h]', 'EngineSpeed[rpm]']])
    df.loc[df['predict_HALF_LOAD_FuelRate[L/h]']<0, 'predict_HALF_LOAD_FuelRate[L/h]'] = 0.

    df['predict_FuelRate[L/h]'] = df.apply(lambda x: x['predict_HALF_LOAD_FuelRate[L/h]'] +
                                           (x['predict_FULL_LOAD_FuelRate[L/h]']-x['predict_HALF_LOAD_FuelRate[L/h]']) /
                                           (VW_FULL_LOAD-VW_HALF_LOAD) *
                                           (x['VehicleWeight[ton]']-VW_HALF_LOAD), axis=1)

    st.subheader(f"累計行駛 {df['VehicleSpeed[km/h]'].sum()/3.6/1000:.2f} 公里")
    st.subheader(f"預測用油 {df['predict_FuelRate[L/h]'].sum()/3600:.2f} 公升")
    st.subheader(f"預測能效 {(df['VehicleSpeed[km/h]'].sum()/3.6/1000)/(df['predict_FuelRate[L/h]'].sum()/3600):.2f} 公里/公升")


def model_6():
    st.markdown("**:green[此處使用日本重車能效法規的車輛空氣阻力係數、輪胎滾動阻力係數等經驗式]**")
    st.write("")
    
    density_deisel = 836. # 柴油密度[kg/m3] @ VECTO
    
    col1, col2 = st.columns(2)
    gvw = col1.number_input("輸入車輛核定總重[噸]", value=26.0, placeholder="Type a number...")
    W0 = col2.number_input("輸入車輛空車重[噸]", value=14.975, placeholder="Type a number...")
    B = col1.number_input("輸入車寬[公尺]", value=2.6, placeholder="Type a number...")
    H = col2.number_input("輸入車高[公尺]", value=3.75, placeholder="Type a number...")
    CC = st.number_input("輸入引擎總排氣量[cc]", value=12913, placeholder="Type a number...")
    
    col3, col4 = st.columns(2)
    i_m_n_r = col3.text_input("輸入變速箱檔位及齒比，格式如{1:14.68, ...}", "{1:14.68, 2:12.05, 3:9.92, 4:8.14, 5:6.78, 6:5.56, 7:4.57, 8:3.75, 9:3.22, 10:2.64, 11:2.17, 12:1.78, 13:1.49, 14:1.22, 15:1, 16:0.82}")
    i_m_1_n = ast.literal_eval(i_m_n_r) # 轉成dict
    i_m_m = list(i_m_1_n.keys())[-1] # 最大的檔位數
    i_f = col4.number_input("輸入差速器減速比", value=3.727, placeholder="Type a number...")
    eff_m = 0.98 # 傳動效率
    eff_f = 0.95
    
    wheel = st.text_input("輸入驅動輪規格，格式如318/80R22.5", "318/80R22.5")
    wheel_D = (float(wheel.split("R")[0].split('/')[0]) *
               float(wheel.split("R")[0].split('/')[1])/100.*2. +
               float(wheel.split("R")[1])*25.4)/1000.
    r = wheel_D / 2.

    st.subheader("")    
    st.write("可自行上傳逐秒之車速、引擎轉速、道路坡度、車重資料檔案(.csv)，需以「,」區隔欄位(columns)。(註: 車重=空車重+載重)")
    st.write("檔案的第一列(row)需有「VehicleWeight[ton], VehicleSpeed[km/h], EngineSpeed[rpm], grad[%]」四欄位名稱, 其餘列為數字資料。")
    st.write("")
    uploaded_file=st.file_uploader("選擇上傳檔案：",type=".csv")    
    st.write("")
    if uploaded_file:
        df=pd.read_csv(uploaded_file, usecols=['VehicleSpeed[km/h]', 'EngineSpeed[rpm]', 'grad[%]', 'VehicleWeight[ton]'])
        st.write("已上傳檔案：", uploaded_file.name)
        st.write(f"上傳的行駛操作資料(共{len(df)}筆紀錄)：")
        df['time[s]'] = [i for i in range(1, len(df)+1)]
        df = df[['time[s]', 'VehicleWeight[ton]', 'VehicleSpeed[km/h]', 'EngineSpeed[rpm]', 'grad[%]']]
    else:
        df=pd.read_csv('default_data.csv')
        st.write(f"預設的行駛操作資料(共{len(df)}筆紀錄)：")  

    st.dataframe(df)    

    # 14.975+(26-14.975)*0.9
    with open('model_6_spacing_12x12_FULL_LOAD.pkl', 'rb') as f:
        reg_FULL_LOAD = pickle.load(f)
    # 14.975+(26-14.975)*0.55
    with open('model_6_spacing_12x12_HALF_LOAD.pkl', 'rb') as f:
        reg_HALF_LOAD = pickle.load(f)
    
    df['acc[m/s^2]'] = (df.loc[0, 'VehicleSpeed[km/h]'] - 0.) / 3.6 # 資料為逐秒紀錄
    for i in range(1, len(df)):
        df.loc[i, 'acc[m/s^2]'] = (df.loc[i, 'VehicleSpeed[km/h]'] - df.loc[i-1, 'VehicleSpeed[km/h]']) / 3.6
    
    # 檔位判斷，忽略空檔、換檔間的操作情況
    def v_to_n(v, g):
        """ 從車速[km/h]、檔位g，計算引擎轉速[rpm] """
        if gear_ratio := i_m_1_n.get(g):
            return (v*1000./60.) / (wheel_D*np.pi) * i_f * gear_ratio
        else:
            print("Gear Position Error!")
    
    def predit_gear_position(v, n):
        """ 從車速[km/h]，計算1~16檔的可能引擎轉速，並取出最接近引擎轉速的檔位 """
        l=[]
        for g in range(1, i_m_m+1):
            l.append(abs(v_to_n(v,g)-n))
        return l.index(min(l))+1
    
    df['predit_gear_position'] = df.apply(lambda x: predit_gear_position(x['VehicleSpeed[km/h]'], x['EngineSpeed[rpm]']), axis=1)
    df['i_m'] = df['predit_gear_position'].apply(lambda x :i_m_1_n.get(x))
    
    
    fig, ax = plt.subplots(figsize=(16, 8))
    norm = matplotlib.colors.Normalize(vmin=1, vmax=i_m_m)
    sc = ax.scatter(df['time[s]'], df['VehicleSpeed[km/h]'], c=df['predit_gear_position'], cmap='viridis', s=20, norm=norm)
    ax.set_xlabel('時間[s]')
    ax.set_ylabel('車速[km/h]')
    plt.title("依車速、引擎轉速判斷變速箱操作檔位(忽略空檔、換檔間的操作情況)")
    plt.colorbar(sc, label='檔位')
    
    st.pyplot(fig)


    df['mu_r[kg/kg]'] = 0.00513 + 17.6 / (df['VehicleWeight[ton]'] * 1000.)
    df['mu_aA[kg/(km/h)2]'] = 0.00299 * B * H - 0.000832
    df['W_eq[kg]'] = (0.07 + 0.03*df['i_m']*df['i_m']) * (W0 * 1000.)
    df['F_rr[N]'] = df['mu_r[kg/kg]'] * (df['VehicleWeight[ton]'] * 1000.) * 9.81 # 調整單位為N
    df['F_slope[N]'] = (df['VehicleWeight[ton]'] * 1000.) * np.sin(np.arctan(df['grad[%]'] / 100)) * 9.81
    df['F_air[N]'] = df['mu_aA[kg/(km/h)2]'] * df['VehicleSpeed[km/h]'] * df['VehicleSpeed[km/h]'] * 9.81
    df['F_acc[N]'] = ((df['VehicleWeight[ton]'] * 1000.) + df['W_eq[kg]']) * df['acc[m/s^2]']
    
    df['R[N]'] = df['F_rr[N]'] + df['F_slope[N]'] + df['F_air[N]'] +df['F_acc[N]']

    
    def engine_torque(R, i_m): # 此R的單位為[N]
        if R>=0: # 原稿為R>0
            return r / eff_m / eff_f / i_m / i_f * R
        else:
            return r * eff_m * eff_f / i_m / i_f * R
    
    df['engine_torque[Nm]'] = df.apply(lambda x: engine_torque(x['R[N]'], x['i_m']), axis=1)
    df['engine_power[kW]'] = df['engine_torque[Nm]'] * (df['EngineSpeed[rpm]'] * 2. * np.pi / 60.) / 1000.


    VW_FULL_LOAD = 14.975+(26-14.975)*0.9 # [ton]
    df['predict_FULL_LOAD_BSFC[g/kWh]'] = reg_FULL_LOAD.predict(df[['EngineSpeed[rpm]', 'engine_torque[Nm]']])
    df['predict_FULL_LOAD_FuelRate[L/h]'] = df['predict_FULL_LOAD_BSFC[g/kWh]'] * df['engine_power[kW]'] / 1000. / density_deisel * 1000.
    df.loc[df['predict_FULL_LOAD_FuelRate[L/h]']<0, 'predict_FULL_LOAD_FuelRate[L/h]'] = 0.
    
    VW_HALF_LOAD = 14.975+(26-14.975)*0.55 # [ton]
    df['predict_HALF_LOAD_BSFC[g/kWh]'] = reg_HALF_LOAD.predict(df[['EngineSpeed[rpm]', 'engine_torque[Nm]']])
    df['predict_HALF_LOAD_FuelRate[L/h]'] = df['predict_HALF_LOAD_BSFC[g/kWh]'] * df['engine_power[kW]'] / 1000. / density_deisel * 1000.
    df.loc[df['predict_HALF_LOAD_FuelRate[L/h]']<0, 'predict_HALF_LOAD_FuelRate[L/h]'] = 0.

    df['predict_FuelRate[L/h]'] = df.apply(lambda x: x['predict_HALF_LOAD_FuelRate[L/h]'] +
                                           (x['predict_FULL_LOAD_FuelRate[L/h]']-x['predict_HALF_LOAD_FuelRate[L/h]']) /
                                           (VW_FULL_LOAD-VW_HALF_LOAD) *
                                           (x['VehicleWeight[ton]']-VW_HALF_LOAD), axis=1)

    st.subheader(f"累計行駛 {df['VehicleSpeed[km/h]'].sum()/3.6/1000:.2f} 公里")
    st.subheader(f"預測用油 {df['predict_FuelRate[L/h]'].sum()/3600:.2f} 公升")
    st.subheader(f"預測能效 {(df['VehicleSpeed[km/h]'].sum()/3.6/1000)/(df['predict_FuelRate[L/h]'].sum()/3600):.2f} 公里/公升")


    # st.dataframe(df[['time[s]', 'VehicleWeight[ton]', 'VehicleSpeed[km/h]', 'EngineSpeed[rpm]', 'grad[%]',
    #                  'F_rr[N]', 'F_slope[N]', 'F_air[N]', 'F_acc[N]',
    #                  'engine_torque[Nm]', 'engine_power[kW]']]) 


    
    
    
    

#網頁的sidebar版面
st.sidebar.header("大貨車行駛及沿途上下貨之操作能效模擬 - v0.11")
st.sidebar.markdown("**:green[以26噸大貨車道路實測數據推估]**")
option=st.sidebar.selectbox("功能模式：",
                            options=['1. 耗油量=f(車速) 依迴歸分析',
                                     '5. 耗油量=f(車速, 引擎轉速) 依迴歸分析',
                                     '6. 制動馬力單位耗油量=f(引擎轉速, 引擎扭矩) 依車輛動力學(逆向動力傳遞)'])
st.sidebar.subheader("")
st.sidebar.write(f"目前選用模式 {option} 推估")
operation={'1. 耗油量=f(車速) 依迴歸分析': model_1,
           '5. 耗油量=f(車速, 引擎轉速) 依迴歸分析': model_5,
           '6. 制動馬力單位耗油量=f(引擎轉速, 引擎扭矩) 依車輛動力學(逆向動力傳遞)': model_6}




#主程式
if __name__ == "__main__":
    operation[option]()



