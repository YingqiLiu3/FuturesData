import numpy as np
import jieba
import tushare as ts
# 初始化pro接口
pro = ts.pro_api('ea3b5e388f7afe70e426f87d5cbeb0796757d219d963e355a67cddfa')

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
    for i in range(0,len(data1)):
        data1[i].append("中金所")
        data.append(data1[i])
    for i in range(0,len(data2)):
        data2[i].append("大商所")
        data.append(data2[i])
    for i in range(0,len(data3)):
        data3[i].append("郑商所")
        data.append(data3[i])
    for i in range(0,len(data4)):
        data4[i].append("上期所")
        data.append(data4[i])
    for i in range(0,len(data5)):
        data5[i].append("上海国际能源交易中心")
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
    df = pro.fut_daily(**{
        "trade_date": "",
        "ts_code": code,
        "exchange": "",
        "start_date": "",
        "end_date": "",
        "limit": 20,
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "pre_close",
        "pre_settle",
        "open",
        "high",
        "low",
        "close",
        "settle",
        "change1",
        "change2",
        "vol",
        "amount",
        "oi",
        "oi_chg",
        "delv_settle"
    ])
    realtime_data = np.array(df).tolist()
    for i in range(0,len(realtime_data)):
        for j in range(0,len(realtime_data[i])):
            realtime_data[i][j]=str(realtime_data[i][j])
    all_data.append(realtime_data)
    return all_data

if __name__ == '__main__':
    print(base_code()[0])