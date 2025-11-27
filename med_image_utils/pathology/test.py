import pandas as pd
import os
from pypinyin import lazy_pinyin


year = '2024年'
df = pd.read_excel(r"C:\Users\Administrator\Desktop\交集唯一项.xlsx", sheet_name=year)

for i in df['姓名'].to_list():
    name_pinyin = ''.join(lazy_pinyin(str(i))).lower()
    os.mkdir(os.path.join(r'C:\Users\Administrator\Desktop\新建文件夹', year, name_pinyin),)