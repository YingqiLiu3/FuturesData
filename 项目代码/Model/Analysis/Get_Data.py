# -*- coding:UTF-8 -*-
import requests
import numpy
import os
# 导入tushare
import tushare as ts
import numpy as np
import pandas as pd
import xmnlp

# 初始化pro接口
pro = ts.pro_api('ea3b5e388f7afe70e426f87d5cbeb0796757d219d963e355a67cddfa')

os.environ['NO_PROXY'] = 'www.cls.cn'

columns_list = ['代码','名称','周报成交量','周报交易量同比增减','周报成交金额','周报成交金额同比增减','周报年累计成交总量','周报年累计成交总量同比增减','周报年累计成交金额','周报年累计成交金额同比增减','周报持仓量','周报持仓量环比增减','本周主力合约收盘价','本周主力合约收盘价环比涨跌','昨日仓单量','今日仓单量','仓单量增减量','成交量','成交量变化','持买仓量','持买仓量变化','持卖仓量','持卖仓量变化','消极情绪','积极情绪']
xmnlp.set_model('./xmnlp-onnx-models-v4/xmnlp-onnx-models')

def nlp(word):
    tag = xmnlp.sentiment(word)
    return tag

def all_code():
    # 拉取数据
    exchange = ["CFFEX","DCE","CZCE","SHFE","INE"]
    data = []
    for index in range(0,len(exchange)):
        df = pro.fut_basic(**{
            "exchange": exchange[index],
            "fut_type": "",
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "name",
            "fut_code"
        ])
        data_tf=np.array(df).tolist()
        for item in data_tf:
            data.append(item)
    code_list = []
    for i in range(0,len(data)):
        ts_code = data[i][0]
        name = data[i][1]
        symbol = data[i][2]
        judge = True
        for j in range(0,len(code_list)):
            if(code_list[j][0]==symbol):
                insert = []
                insert.append(ts_code)
                insert.append(name)
                code_list[j][1].append(insert)
                judge=False
                break
        if(judge):
            insert=[]
            insert.append(ts_code)
            insert.append(name)
            insert_need=[]
            insert_need.append(insert)
            code_data=[]
            code_data.append(symbol)
            code_data.append(insert_need)
            code_list.append(code_data)
    return code_list

def get_all_data():
    all_co = all_code()
    all_data = []
    for index in range(0,len(all_co)):
        symbol_data = []
        symbol_data.append(all_co[index][0])
        all_data.append(symbol_data)
    for index in range(0,len(all_data)):
        symbol = all_data[index][0]
        df = pro.fut_weekly_detail(**{"prd": symbol,"limit": 20}, fields=["name","vol","vol_yoy","amount","amout_yoy","cumvol","cumvol_yoy","cumamt","cumamt_yoy","open_interest","interest_wow","mc_close","close_wow"])
        data = np.array(df).tolist()
        print(symbol)
        if(len(data)==0):
            all_data[index].append("NoneName")
        else:
            all_data[index].append(data[0][0])
        for i in range(1,13):
            sum=0
            num=0
            for j in range(0,len(data)):
                if(str(data[j][i])!="nan" and str(data[j][i])!="None"):
                    sum+=data[j][i]
                    num+=1
            if(num==0):
                all_data[index].append(0.0)
            else:
                all_data[index].append(sum/num)


    for index in range(0,len(all_data)):
        symbol = all_data[index][0]
        # 拉取数据
        df = pro.fut_wsr(**{"symbol": symbol}, fields=["trade_date","fut_name","pre_vol","vol","vol_chg"])
        data = np.array(df).tolist()
        print(symbol)
        day_data = []
        for i in range(0,len(data)):
            day = data[i][0]
            flag = 0
            if (str(data[i][2]) == "nan" or str(data[i][2]) == "None"):
                data[i][2] = 0
            if (str(data[i][3]) == "nan" or str(data[i][3]) == "None"):
                data[i][3] = 0
            if (str(data[i][4]) == "nan" or str(data[i][4]) == "None"):
                data[i][4] = 0
            while(flag<len(day_data)):
                if(day == day_data[flag][0]):
                    day_data[flag][1][0] += data[i][2]
                    day_data[flag][1][1] += data[i][3]
                    day_data[flag][1][2] += data[i][4]
                    break
                else:
                    flag+=1
            if(flag==len(day_data)):
                if(len(day_data)==20):
                    break
                insert_data = []
                insert_data.append(day)
                insert_data.append(data[i][2:])
                day_data.append(insert_data)
        for i in range(0,3):
            sum=0
            num=0
            for j in range(0,len(day_data)):
                if(str(day_data[j][1][i])!="nan" and str(day_data[j][1][i])!="None"):
                    sum+=day_data[j][1][i]
                    num+=1
            if(num==0):
                all_data[index].append(0.0)
            else:
                all_data[index].append(sum/num)
    for index in range(0,len(all_data)):
        symbol = all_data[index][0]
        df = pro.fut_holding(**{"symbol": symbol}, fields=["trade_date","vol","vol_chg","long_hld","long_chg","short_hld","short_chg"])
        data = np.array(df).tolist()
        print(symbol)
        day_data = []
        for i in range(0,len(data)):
            day = data[i][0]
            flag = 0
            if (str(data[i][1]) == "nan" or str(data[i][1]) == "None"):
                data[i][1] = 0
            if (str(data[i][2]) == "nan" or str(data[i][2]) == "None"):
                data[i][2] = 0
            if (str(data[i][3]) == "nan" or str(data[i][3]) == "None"):
                data[i][3] = 0
            if (str(data[i][4]) == "nan" or str(data[i][4]) == "None"):
                data[i][4] = 0
            if (str(data[i][5]) == "nan" or str(data[i][5]) == "None"):
                data[i][5] = 0
            if (str(data[i][6]) == "nan" or str(data[i][6]) == "None"):
                data[i][6] = 0
            while(flag<len(day_data)):
                if(day == day_data[flag][0]):
                    day_data[flag][1][0] += data[i][1]
                    day_data[flag][1][1] += data[i][2]
                    day_data[flag][1][2] += data[i][3]
                    day_data[flag][1][3] += data[i][4]
                    day_data[flag][1][4] += data[i][5]
                    day_data[flag][1][5] += data[i][6]
                    break
                else:
                    flag+=1
            if(flag==len(day_data)):
                if(len(day_data)==20):
                    break
                insert_data = []
                insert_data.append(day)
                insert_data.append(data[i][1:])
                day_data.append(insert_data)
        for i in range(0,6):
            sum=0
            num=0
            for j in range(0,len(day_data)):
                if(str(day_data[j][1][i])!="nan" and str(day_data[j][1][i])!="None"):
                    sum+=day_data[j][1][i]
                    num+=1
            if(num==0):
                all_data[index].append(0.0)
            else:
                all_data[index].append(sum/num)
    return all_data

def stock_zh_a_alerts_cls(word) -> pd.DataFrame:
    """
    财联社-今日快讯
    https://www.cls.cn/searchPage?keyword=%E5%BF%AB%E8%AE%AF&type=all
    :return: 财联社-今日快讯
    :rtype: pandas.DataFrame
    """
    url = "https://www.cls.cn/api/sw"
    params = {"app": "CailianpressWeb", "os": "web", "sv": "7.5.5"}
    headers = {
        "Host": "www.cls.cn",
        "Connection": "keep-alive",
        "Content-Length": "112",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": "https://www.cls.cn",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    payload = {
        "type": "telegram",
        "keyword": word,
        "page": 0,
        "rn": 10000,
        "os": "web",
        "sv": "7.2.2",
        "app": "CailianpressWeb",
    }
    r = requests.post(url, headers=headers, params=params, json=payload)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["data"]["telegram"]["data"])
    if(len(temp_df)==0):
        return []
    temp_df = temp_df[["descr", "time"]]
    temp_df["descr"] = temp_df["descr"].astype(str).str.replace("</em>", "")
    temp_df["descr"] = temp_df["descr"].str.replace("<em>", "")
    temp_df["time"] = pd.to_datetime(temp_df["time"], unit="s").dt.date
    temp_df.columns = ["快讯信息", "时间"]
    temp_df = temp_df[["时间", "快讯信息"]]
    return temp_df

def add_info():
    data = get_all_data()
    for index in range(0,len(data)):
        text = data[index][1]
        print(text)
        df = np.array(stock_zh_a_alerts_cls(text))
        if(len(df)==0):
            data[index].append(0.0)
            data[index].append(0.0)
        else:
            need = [0.0,0.0]
            for i in range(0, len(df)):
                word = df[i][1]
                item = nlp(word)
                need[0]+=item[0]
                need[1]+=item[1]
            data[index].append(need[0]/len(df))
            data[index].append(need[1]/len(df))
    return data

if __name__ == '__main__':
    data = add_info()
    np.save("./Data/data_end.npy",np.array(data))
    df = pd.DataFrame(data,columns=columns_list)
    df.to_csv("./Data/data.csv",index=False,encoding="utf_8_sig")
    print(df)