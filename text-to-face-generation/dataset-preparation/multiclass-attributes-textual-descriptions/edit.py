import pandas as pd
import random
table = pd.read_csv('./attributes.csv')
for i, row in table.iterrows():
    print(i)
    if row.Wearing_SightGlasses==1 or row.Wearing_SunGlasses==1:
        if random.random() > 0.5:
            table.at[i,'Wearing_SightGlasses'] = 1
            table.at[i,'Wearing_SunGlasses'] = 0
        else:
            table.at[i,'Wearing_SightGlasses'] = 0
            table.at[i,'Wearing_SunGlasses'] = 1
table.to_csv('attributes.csv')
