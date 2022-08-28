
import streamlit as st
import plotly.graph_objects as go
import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly


START = "2015-01-02"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOGL', 'AAPL', 'MSFT', 'GME','TECHM.NS')
selected_stock = st.selectbox('Select Stock', stocks)


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Open and Close
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data of Stock',xaxis_title='Date',
    yaxis_title='Price',xaxis_showgrid=False, yaxis_showgrid=False, xaxis_rangeslider_visible=True,autosize=False,
    width=1200,
    height=800)
	st.plotly_chart(fig)

plot_raw_data()


#st.subheader('Stock price with MA100 and MA200')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig2=go.Figure()
fig2.add_trace(go.Scatter(x=data.Date, y=data.Close, line=dict(color='deepskyblue', width=1),name='Close Price'))
fig2.add_trace(go.Scatter(x=data.Date, y=ma100, line=dict(color='firebrick', width=1),name='MA100'))
fig2.add_trace(go.Scatter(x=data.Date, y=ma200, line=dict(color='darkseagreen', width=1),name='MA200'))
fig2.layout.update(title_text='Stock price with Moving Average 100 & 200',xaxis_title='Date',
    yaxis_title='Price',xaxis_showgrid=False, yaxis_showgrid=False, xaxis_rangeslider_visible=True,autosize=False,
    width=1200,
    height=800)
st.plotly_chart(fig2)


# Candle wick with MA
fig4 = go.Figure(data=[go.Candlestick(x=data.Date,
                                     open=data.Open, 
                                     high=data.High,
                                     low=data.Low,
                                     close=data.Close,
                                     name='Candle Wick'), 
                      go.Scatter(x=data.Date, y=ma100, line=dict(color='slategrey', width=1),name='MA100'),
                      go.Scatter(x=data.Date, y=ma200, line=dict(color='darkolivegreen', width=1),name='MA200')])
fig4.update_layout(
    title='Candle wick with Moving Average',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_showgrid=False, yaxis_showgrid=False,autosize=False,
    width=1200,
    height=800
)
st.plotly_chart(fig4)

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast Data')
st.write(forecast.tail())
    
st.subheader('Forecast Plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
