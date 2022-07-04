import streamlit as st

st.set_page_config(page_title = 'Stocker', page_icon = ':chart_with_upwards_trend:' , layout = 'wide')

st.subheader('This is...')
st.title('Stocker')
st.write('เว็บแอปทำนายราคาปิดหุ้นในอนาคตโดยใช้ปัญญาประดิษฐ์:sunglasses:')

st.write('---')

st.header('เลือกข้อมูล')
st.write('##')
name = st.text_input('หุ้นชื่อย่ออะไร: ', 'PTTGC.BK')
date1 = st.text_input('เรียกดูข้อมูลตั้งแต่วันไหน: ', '2022-01-01')
date2 = st.text_input('เรียกดูข้อมูลจนถึงวันไหน: ', '2022-06-30')
date3 = st.text_input('ทำนายราคาปิดจนถึงวันไหน: ', '2022-07-31')

st.write('---')

st.header('ปรับพารามิเตอร์')
st.write('##')
day = st.number_input('ทำนายราคาปิดแต่ละวันโดยใช้ข้อมูลย้อนหลังทั้งหมดกี่วัน: ', min_value = 1, value = 30)
percent = st.number_input('ข้อมูลที่เรียกดูทั้งหมดจะแบ่งมาใช้ในการเทรนกี่เปอร์เซ็นต์: ', min_value = 1, max_value = 100, value = 80)

st.write('---')

with st.spinner('แปป...'):

    import yfinance as yf
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from sklearn.preprocessing import MinMaxScaler
    plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

    input = day
    dense = math.ceil(((2 / 3) * input) + 1)
    output = 1
    ratio = percent / 100
    optimizer = 'nadam'
    loss = 'mean_squared_error'
    
    @st.experimental_memo
    def load_model():
        return yf.download(name, date1, date2)
    
    df0 = load_model()

    df1 = df0.filter(['Close'])
    le0 = len(df1)

    date_list = [i.strftime('%Y-%m-%d 00:00:00') for i in pd.date_range(date2, date3, freq = 'B')]
    df2 = pd.DataFrame(date_list)
    df2.rename(columns = {df2.columns[0]: 'Date'}, inplace = True)
    df2['Predictions'] = np.nan
    df2.set_index('Date', inplace = True)
    df2.index = pd.to_datetime(df2.index)

    df3 = pd.concat([df1, df2])
    df3 = df3.fillna(np.nan)

    df4 = df3.filter(['Close'])
    ra1 = df4.values
    le1 = math.ceil(le0 * ratio)

    df3.reset_index(inplace = True)

    scaler = MinMaxScaler(feature_range = (0,1))
    sc1 = scaler.fit_transform(ra1)

    tr0 = sc1[0:le1,:]
    tr1 = []
    tr2 = []
    for i in range(day, le1):
        tr1.append(tr0[i-day:i,0])
        tr2.append(tr0[i,0])

    tr1, tr2 = np.array(tr1), np.array(tr2)
    tr1 = np.reshape(tr1, (tr1.shape[0], tr1.shape[1], 1))

    model = Sequential()
    model.add(LSTM(input, return_sequences = True, input_shape = (tr1.shape[1], 1)))
    model.add(LSTM(input, return_sequences = False))
    model.add(Dense(dense))
    model.add(Dense(output))

    model.compile(optimizer, loss)

    model.fit(tr1, tr2, batch_size = 1, epochs = 3)

    te0 = sc1[le1 - day:,:]
    le2 = len(te0)
    le = len(sc1)
    te1 = []
    te1.append(sc1[le1 - day:le1,0])

    te1 = np.array(te1)

    te1 = np.reshape(te1, (te1.shape[0], te1.shape[1], 1))

    for i in range(le1 + 1, le):
        re = model.predict(te1)
        ta = scaler.inverse_transform(re)
        ta = ta[0].tolist()[0]
        df3.loc[i, 'Predictions'] = ta
        re = re.tolist().pop(0).pop(0)
        te1 = te1.tolist().pop(0)
        te3 = []
        for j in range(day):
            te3.append(te1.pop(0).pop(0))
        te3.pop(0)
        te3.append(re)
        te4 = []
        te4.append(te3[:])
        te1 = te4
        te1 = np.array(te1)
        te1 = np.reshape(te1, (te1.shape[0], te1.shape[1], 1))

    te2 = ra1[le1:le0,:]

    df5 = df3

    df3.set_index('Date',inplace = True)
    df3.index = pd.to_datetime(df3.index)

    df6 = df5.iloc[le1 + 1:,:]

    da1 = df3.filter(['Close'])
    da2 = df3.filter(['Predictions'])
    li1 = da1[:le1]
    li2 = da1[le1:le0]
    li3 = da2[le1:le0]
    li4 = da2[le0:]

    rmse1 = 0
    for i in range(le1 + 1, le0):
        pop1 = df5.filter(['Close']).values.tolist()[i].pop(0)
        pop2 = df5.filter(['Predictions']).values.tolist()[i].pop(0)
        rmse0 = (pop1 - pop2) ** 2
        rmse1 = rmse1 + rmse0
    if le0 - le1 - 1 == 0:
        rmse = 'undefined'
    else:
        rmse2 = rmse1 / (le0 - le1 - 1)
        rmse = rmse2 ** 0.5

    fig = plt.figure(figsize = (20,8))
    plt.title(name, fontsize = 18)
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price THB', fontsize = 18)
    plt.plot(li1['Close'], c = 'violet')
    plt.plot(li2['Close'], c = 'darkviolet')
    plt.plot(li3['Predictions'], c = 'green')
    plt.plot(li4['Predictions'], c = 'limegreen')
    plt.legend(['Real Data (Training)', 'Real Data (Testing)', 'Prediction (Testing)', 'Prediction (Future)'], loc = 'lower left')

st.success('เสร็จละ')

st.header('ผลการทำนายราคาปิดหุ้นรายวัน')
st.write('##')

st.write(fig)

st.write(df6)

st.write('Test Root Mean Square Error: ')
st.write(rmse)

st.write('---')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
local_css("style/style.css")

st.header('มาคุยกัน')
st.write('##')

contact_form = '''
<form action = 'https://formsubmit.co/44cb3e677835b6d430f0e09576b4dece' method = 'POST'>
     <input type = 'text' name = 'name' placeholder = 'ชื่อ' required>
     <input type = 'email' name = 'email' placeholder = 'อีเมล' required>
     <textarea name = 'message' placeholder = 'ข้อความ' required></textarea>
     <button type="submit">Send</button>
</form>
'''

st.markdown(contact_form, unsafe_allow_html = True)

st.experimental_memo.clear()
