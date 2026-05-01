import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

df = pd.read_csv('ecommerce_sales_analytics_5000.csv')
df['order_date'] = pd.to_datetime(df['order_date'])

df = df.rename(columns={
    'order_id': 'Order_ID',
    'order_date': 'Order_Date',
    'customer_id': 'CustomerID',
    'product_category': 'Category',
    'quantity': 'Quantity',
    'unit_price': 'Price_Per_Unit',
    'revenue': 'Total_Sales'
})

category_budget = {'Beauty': 5000, 'Clothing': 7000, 'Electronics': 9000, 'Home': 4000}
df['Ad_Budget'] = df['Category'].map(category_budget)

product = df.groupby('Category').agg({
    'Price_Per_Unit': 'mean',
    'Quantity': 'sum',
    'Total_Sales': 'sum'
}).reset_index()

snapshot_date = df['Order_Date'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'Order_Date': lambda x: (snapshot_date - x.max()).days,
    'Order_ID': 'count',
    'Total_Sales': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

category = df.groupby('Category').agg({
    'Total_Sales': 'sum',
    'Ad_Budget': 'first'
}).reset_index()

category['Efisiensi'] = category['Total_Sales'] / category['Ad_Budget']
category = category.sort_values('Efisiensi')

median_budget = df['Ad_Budget'].median()
high_ad = df[df['Ad_Budget'] >= median_budget]
low_ad = df[df['Ad_Budget'] < median_budget]

t_stat, p_value = stats.ttest_ind(high_ad['Total_Sales'], low_ad['Total_Sales'])

def categorize(score):
    r, f, m = int(score[0]), int(score[1]), int(score[2])
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 4 and m >= 4:
        return 'Loyal'
    elif m >= 4:
        return 'Big Spender'
    elif r <= 2 and m >= 4:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Others'

rfm['Kategori'] = rfm['RFM_Segment'].apply(categorize)

X = df[['Ad_Budget']].values
y = df['Total_Sales'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)
