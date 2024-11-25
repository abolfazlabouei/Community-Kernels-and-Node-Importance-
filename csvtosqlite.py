import pandas as pd 
import sqlite3

conn = sqlite3.connect("subscription.db")

df = pd.read_csv('subscription.csv')

df.to_sql('my_table', conn, if_exists='replace', index=False)
print(pd.read_sql('SELECT * FROM my_table', conn))

