import akshare as ak
import numpy as np
import jieba
# 导入tushare
import tushare as ts
import datetime
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import h5py

# 初始化pro接口
pro = ts.pro_api('ea3b5e388f7afe70e426f87d5cbeb0796757d219d963e355a67cddfa')

def all_code():
    time_str = datetime.datetime.now().strftime('%Y%m%d')
    last_work_time = ""

    # 拉取数据
    time_df = pro.trade_cal(**{
        "end_date": time_str
    }, fields=[
        "cal_date",
        "is_open"
    ])
    time_data = np.array(time_df).tolist()[::-1]
    for i in range(0, len(time_data)):
        if (time_data[i][1] == 1):
            last_work_time = time_data[i][0]
            break
    df = pro.fut_daily(**{
        "trade_date": last_work_time
    }, fields=[
        "ts_code"
    ])
    return np.array(df).tolist()

# 将数据截取成3个一组的监督学习格式
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_lstm(data,name):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 定义随机种子，以便重现结果
    np.random.seed(7)
    # 加载数据
    dataset = data.astype('float32')
    # 缩放数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # 分割2/3数据作为测试
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    '''
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    '''
    need = dataset
    # 预测数据步长为3,三个预测一个，3->1
    look_back = 3
    '''
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    '''
    needX, needY = create_dataset(need, look_back)
    if(len(needX)==0):
        return
    # 重构输入数据格式 [samples, time steps, features] = [93,3,1]
    '''
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    '''
    needX = np.reshape(needX, (needX.shape[0], needX.shape[1], 1))
    # 构建 LSTM 网络
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(needX, needY, epochs=50, batch_size=batch_size, verbose=2, shuffle=False)
    # 保存模型
    model.save("./Model_Save/"+name+".h5")
    '''
    # 对训练数据的Y进行预测
    trainPredict = model.predict(trainX, batch_size=batch_size)
    # 对测试数据的Y进行预测
    testPredict = model.predict(testX, batch_size=batch_size)
    # 对数据进行逆缩放
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # 计算RMSE误差
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # 构造一个和dataset格式相同的数组，共145行，dataset为总数据集，把预测的93行训练数据存进去
    trainPredictPlot = np.empty_like(dataset)
    # 用nan填充数组
    trainPredictPlot[:, :] = np.nan
    # 将训练集预测的Y添加进数组，从第3位到第93+3位，共93行
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # 构造一个和dataset格式相同的数组，共145行，把预测的后44行测试数据数据放进去
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    # 将测试集预测的Y添加进数组，从第94+4位到最后，共44行
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    '''
    """
    # 画图
    plt.plot(scaler.inverse_transform(dataset), label='数据集')
    plt.plot(trainPredictPlot, label='预测值')
    plt.plot(testPredictPlot, label='真实值')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.title("LSTM预测")
    plt.xlabel("时间轴")
    plt.ylabel("预测")
    plt.show()
    """

def make_data(code):
    df = pro.fut_daily(**{
        "ts_code": code,
        "limit": 200
    }, fields=[
        "open",
        "high",
        "low",
        "close",
        "settle",
        "trade_date"
    ])
    return np.array(df).tolist()[::-1]

def get_data(code):
    data = make_data(code)
    data_open = []
    data_high = []
    data_low = []
    data_close = []
    data_settle = []
    for i in range(0,len(data)):
        if(str(data[i][1])=='nan' or str(data[i][2])=='nan' or str(data[i][3])=='nan' or str(data[i][4])=='nan' or str(data[i][5])=='nan'):
            continue
        data_arr = []
        data_arr.append(data[i][1])
        data_open.append(data_arr)
        data_arr = []
        data_arr.append(data[i][2])
        data_high.append(data_arr)
        data_arr = []
        data_arr.append(data[i][3])
        data_low.append(data_arr)
        data_arr = []
        data_arr.append(data[i][4])
        data_close.append(data_arr)
        data_arr = []
        data_arr.append(data[i][5])
        data_settle.append(data_arr)
    arr_open = np.array(data_open)
    arr_high = np.array(data_high)
    arr_low = np.array(data_low)
    arr_close = np.array(data_close)
    arr_settle = np.array(data_settle)
    return arr_open,arr_high,arr_low,arr_close,arr_settle


def train_one(code):
    arr_open, arr_high, arr_low, arr_close,arr_settle=get_data(code)
    print("train open:"+code)
    train_lstm(arr_open, str(code) + "_" + "open")
    print("train high:" + code)
    train_lstm(arr_high, str(code) + "_" + "high")
    print("train low:" + code)
    train_lstm(arr_low, str(code) + "_" + "low")
    print("train close:" + code)
    train_lstm(arr_close, str(code) + "_" + "close")
    print("train settle:" + code)
    train_lstm(arr_settle, str(code) + "_" + "settle")


def train_all():
    code = all_code()
    print(len(code))
    judge = False
    for i in range(0,len(code)):
        if(code[i][0]=="RB2203.SHF"):
            judge=True
        if(judge):
            train_one(code[i][0])

if __name__ == '__main__':
    train_all()