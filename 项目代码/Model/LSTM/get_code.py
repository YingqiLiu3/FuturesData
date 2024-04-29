import numpy as np
# 导入tushare
import tushare as ts
import datetime
import os.path

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

if __name__ == '__main__':
    data = all_code()
    code = []
    for i in range(0,len(data)):
        symbol = data[i][0]
        path_open = "./Model_Save/" + symbol + "_open.h5"
        path_high = "./Model_Save/" + symbol + "_high.h5"
        path_low = "./Model_Save/" + symbol + "_low.h5"
        path_close = "./Model_Save/" + symbol + "_close.h5"
        path_settle = "./Model_Save/" + symbol + "_settle.h5"
        if(os.path.isfile(path_open) and os.path.isfile(path_high) and os.path.isfile(path_low) and os.path.isfile(path_close) and os.path.isfile(path_settle)):
            code.append(symbol)
    print(code)
    print(len(code))
    np.save("ALL_CODE.npy",np.array(code))