# _*_ coding:utf-8 _*_
import numpy as np
import jieba
import tushare as ts
import matplotlib.pyplot as plt
import tensorflow.keras as tk
from sklearn.preprocessing import MinMaxScaler
import uuid
import base64

pic_url = "https://pic.cpolar.cn/"

# 初始化pro接口
pro = ts.pro_api('ea3b5e388f7afe70e426f87d5cbeb0796757d219d963e355a67cddfa')

last_df = pro.fut_daily(**{
    "ts_code": "A.DCE",
    "limit": 1
}, fields=[
    "trade_date"
])
last_work_time = np.array(last_df)[0][0]
next_df = pro.trade_cal(**{
    "start_date": last_work_time,
    "is_open": 1,
    "limit": 2
}, fields=[
    "cal_date"
])
next_work_time = np.array(next_df)[1][0]

def all_code():
    data = np.load("./Model/LSTM/ALL_CODE.npy")
    return data.tolist()

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
    need = dataset
    # 预测数据步长为3,三个预测一个，3->1
    look_back = 3
    needX, needY = create_dataset(need, look_back)
    needX = np.reshape(needX, (needX.shape[0], needX.shape[1], 1))
    # 构建 LSTM 网络
    lstm_model = tk.models.load_model("./Model/LSTM/Model_Save/"+name+".h5")
    Predict = lstm_model.predict(needX, batch_size=1)
    shape_list = Predict.shape
    needY = needY.reshape(shape_list[0],shape_list[1])
    Predict = scaler.inverse_transform(Predict)
    needY = scaler.inverse_transform(needY)
    # 画图
    plt.plot(needY, label='真实值')
    plt.plot(Predict, label='预测值')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.title("LSTM预测-"+name)
    plt.xlabel("时间轴")
    plt.ylabel("预测")
    uuid_str = uuid.uuid4().hex
    file_name = name+"_"+uuid_str+".png"
    plt.savefig("./Pic/" + file_name)
    plt.close('all')
    forecast_data = []
    insert_data = []
    for i in range(0,3):
        insert = []
        insert.append(dataset[len(dataset)-4+i][0])
        insert_data.append(insert)
    forecast_data.append(insert_data)
    return_data = lstm_model.predict(np.array(forecast_data), batch_size=1)
    return_data = scaler.inverse_transform(return_data)
    ans = []
    ans.append(pic_url+file_name)
    ans.append(str(return_data[0][0]))
    return ans

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
    open_ans = train_lstm(arr_open, str(code) + "_" + "open")
    print("train high:" + code)
    high_ans = train_lstm(arr_high, str(code) + "_" + "high")
    print("train low:" + code)
    low_ans = train_lstm(arr_low, str(code) + "_" + "low")
    print("train close:" + code)
    close_ans = train_lstm(arr_close, str(code) + "_" + "close")
    print("train settle:" + code)
    settle_ans = train_lstm(arr_settle, str(code) + "_" + "settle")
    ans = []
    ans.append(last_work_time)
    ans.append(next_work_time)
    ans.append(open_ans)
    ans.append(high_ans)
    ans.append(low_ans)
    ans.append(close_ans)
    ans.append(settle_ans)
    return ans

def base_code():
    # 拉取数据
    df1 = pro.fut_basic(**{
        "exchange": "CFFEX",
        "fut_type": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "fut_code",
        "multiplier",
        "trade_unit",
        "per_unit",
        "quote_unit",
        "quote_unit_desc",
        "d_mode_desc",
        "list_date",
        "delist_date",
        "d_month",
        "last_ddate",
        "trade_time_desc"
    ])
    data1 = np.array(df1).tolist()
    df2 = pro.fut_basic(**{
        "exchange": "DCE",
        "fut_type": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "fut_code",
        "multiplier",
        "trade_unit",
        "per_unit",
        "quote_unit",
        "quote_unit_desc",
        "d_mode_desc",
        "list_date",
        "delist_date",
        "d_month",
        "last_ddate",
        "trade_time_desc"
    ])
    data2 = np.array(df2).tolist()
    df3 = pro.fut_basic(**{
        "exchange": "CZCE",
        "fut_type": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "fut_code",
        "multiplier",
        "trade_unit",
        "per_unit",
        "quote_unit",
        "quote_unit_desc",
        "d_mode_desc",
        "list_date",
        "delist_date",
        "d_month",
        "last_ddate",
        "trade_time_desc"
    ])
    data3 = np.array(df3).tolist()
    df4 = pro.fut_basic(**{
        "exchange": "SHFE",
        "fut_type": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "fut_code",
        "multiplier",
        "trade_unit",
        "per_unit",
        "quote_unit",
        "quote_unit_desc",
        "d_mode_desc",
        "list_date",
        "delist_date",
        "d_month",
        "last_ddate",
        "trade_time_desc"
    ])
    data4 = np.array(df4).tolist()
    df5 = pro.fut_basic(**{
        "exchange": "INE",
        "fut_type": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "fut_code",
        "multiplier",
        "trade_unit",
        "per_unit",
        "quote_unit",
        "quote_unit_desc",
        "d_mode_desc",
        "list_date",
        "delist_date",
        "d_month",
        "last_ddate",
        "trade_time_desc"
    ])
    data5 = np.array(df5).tolist()
    data = []
    code = all_code()
    for i in range(0,len(data1)):
        data1[i].append("中金所")
        if(data1[i][0] in code):
            data.append(data1[i])
    for i in range(0,len(data2)):
        data2[i].append("大商所")
        if (data2[i][0] in code):
            data.append(data2[i])
    for i in range(0,len(data3)):
        data3[i].append("郑商所")
        if (data3[i][0] in code):
            data.append(data3[i])
    for i in range(0,len(data4)):
        data4[i].append("上期所")
        if (data4[i][0] in code):
            data.append(data4[i])
    for i in range(0,len(data5)):
        data5[i].append("上海国际能源交易中心")
        if (data5[i][0] in code):
            data.append(data5[i])
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            data[i][j]=str(data[i][j])
    return data

def cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    :param sentence1: s
    :param sentence2:
    :return: 两句文本的相识度
    """
    seg1 = [word for word in jieba.cut(sentence1)]
    seg2 = [word for word in jieba.cut(sentence2)]
    word_list = list(set([word for word in seg1 + seg2]))#建立词库
    word_count_vec_1 = []
    word_count_vec_2 = []
    for word in word_list:
        word_count_vec_1.append(seg1.count(word))#文本1统计在词典里出现词的次数
        word_count_vec_2.append(seg2.count(word))#文本2统计在词典里出现词的次数

    vec_1 = np.array(word_count_vec_1)
    vec_2 = np.array(word_count_vec_2)
    #余弦公式

    num = vec_1.dot(vec_2.T)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def search_code(word):
    data = base_code()
    for i in range(0, len(data)):
        data[i].append(str(data[i][0]) + ' ' + str(data[i][1]) + ' ' + str(data[i][2]) + ' ' + str(data[i][3]) + ' ' + str(data[i][16]))
    for i in range(0, len(data)):
        test = cosine_similarity(word, data[i][17])
        data[i][17] = str(test)
    answer = sorted(data, key=lambda s: float(s[17]), reverse=True)
    return answer

def get_all_data(code):
    data = base_code()
    all_data = []
    for i in range(0,len(data)):
        if(data[i][0]==code):
            all_data.append(data[i])
            break
    fore_data = train_one(code)
    all_data.append(fore_data)
    return all_data