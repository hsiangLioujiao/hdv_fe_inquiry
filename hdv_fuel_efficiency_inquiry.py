# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:06:51 2025

@author: g_s_s
"""

import numpy as np
import pandas as pd
import streamlit as st
import pickle


pd.options.mode.copy_on_write = True
st.set_page_config(
    page_title="功能打樣版 僅供3人同時使用",
    page_icon="random",
)


def model_1():
    st.write("可自行上傳逐秒車速[公里/小時]、車重[噸]之.csv資料(註: 車重=空車重+載重)。")
    st.write("所上傳資料需均為數字，不含上述的欄位名稱。")  
    uploaded_file=st.file_uploader("選一檔案",type=".csv")    
    st.header("")
    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        st.write("上傳的檔案名稱：", uploaded_file.name)
        st.write(f"上傳的行駛操作資料(共{len(df)}筆紀錄)：")
    else:
        df=pd.read_csv('model_1_spacing_5km_default_data.csv')
        st.write(f"預設的行駛操作資料(共{len(df)}筆紀錄)：")   

    df.columns=['VehicleSpeed[km/h]', 'VehicleWeight[ton]']
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
    
    st.subheader(f"使用燃油{df['predict_FuelRate[L/h]'].sum()/3600:.2f}公升")
    st.subheader(f"累計行駛{df['VehicleSpeed[km/h]'].sum()/3.6/1000:.2f}公里")
    st.subheader(f"預測能效{(df['VehicleSpeed[km/h]'].sum()/3.6/1000)/(df['predict_FuelRate[L/h]'].sum()/3600):.2f}公里/公升")


def model_5():
    st.write("可自行上傳逐秒車速[公里/小時]、引擎轉速[rpm]、車重[噸]之.csv資料(註: 車重=空車重+載重)。")
    st.write("所上傳資料需均為數字，不含上述的欄位名稱。")  
    uploaded_file=st.file_uploader("選一檔案",type=".csv")    
    st.header("")
    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        st.write("上傳的檔案名稱：", uploaded_file.name)
        st.write(f"上傳的行駛操作資料(共{len(df)}筆紀錄)：")
    else:
        df=pd.read_csv('model_5_spacing_6x6_default_data.csv')
        st.write(f"預設的行駛操作資料(共{len(df)}筆紀錄)：")  

    df.columns=['VehicleSpeed[km/h]', 'EngineSpeed[rpm]', 'VehicleWeight[ton]']
    st.dataframe(df)

    # 14.975+(26-14.975)*0.9
    with open('model_5_spacing_6x6_FULL_LOAD.pkl', 'rb') as f:
        reg_FULL_LOAD = pickle.load(f)
    # 14.975+(26-14.975)*0.55
    with open('model_5_spacing_6x6_HALF_LOAD.pkl', 'rb') as f:
        reg_HALF_LOAD = pickle.load(f)

    df['VW_FULL_LOAD[ton]'] = 14.975+(26-14.975)*0.9
    df['VW_HALF_LOAD[ton]'] = 14.975+(26-14.975)*0.55
    df['predict_FULL_LOAD_FuelRate[L/h]'] = reg_FULL_LOAD.predict(df[['VehicleSpeed[km/h]', 'EngineSpeed[rpm]']])
    df.loc[df['predict_FULL_LOAD_FuelRate[L/h]']<0, 'predict_FULL_LOAD_FuelRate[L/h]'] = 0.
    df['predict_HALF_LOAD_FuelRate[L/h]'] = reg_HALF_LOAD.predict(df[['VehicleSpeed[km/h]', 'EngineSpeed[rpm]']])
    df.loc[df['predict_HALF_LOAD_FuelRate[L/h]']<0, 'predict_HALF_LOAD_FuelRate[L/h]'] = 0.

    df['predict_FuelRate[L/h]'] = df.apply(lambda x: x['predict_HALF_LOAD_FuelRate[L/h]'] +
                                           (x['predict_FULL_LOAD_FuelRate[L/h]']-x['predict_HALF_LOAD_FuelRate[L/h]']) /
                                           (x['VW_FULL_LOAD[ton]']-x['VW_HALF_LOAD[ton]']) *
                                           (x['VehicleWeight[ton]']-x['VW_HALF_LOAD[ton]']), axis=1)

    st.subheader(f"使用燃油{df['predict_FuelRate[L/h]'].sum()/3600:.2f}公升")
    st.subheader(f"累計行駛{df['VehicleSpeed[km/h]'].sum()/3.6/1000:.2f}公里")
    st.subheader(f"預測能效{(df['VehicleSpeed[km/h]'].sum()/3.6/1000)/(df['predict_FuelRate[L/h]'].sum()/3600):.2f}公里/公升")


def model_6():
    pass



#網頁的sidebar版面
st.sidebar.header("大貨車行駛及沿途上下貨之操作能效模擬 - v0.11")
st.sidebar.subheader("以26噸大貨車實測數據推估：")
option=st.sidebar.selectbox("功能模式：",
                            options=['1) 能效=f(車速)',
                                     '5) 能效=f(車速, 引擎轉速)',
                                     '6) 依車輛動力學(逆向動力傳遞)之BSFC=f(引擎轉速, 引擎扭矩)'])
st.sidebar.write("目前選用的功能模式為：", option)
operation={'1) 能效=f(車速)': model_1,
           '5) 能效=f(車速, 引擎轉速)': model_5,
           '6) 依車輛動力學(逆向動力傳遞)之BSFC=f(引擎轉速, 引擎扭矩)': model_6}




#主程式
if __name__ == "__main__":
    operation[option]()



