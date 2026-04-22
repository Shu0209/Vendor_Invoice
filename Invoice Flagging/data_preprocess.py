import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_data(db_path:str):
    conn=sqlite3.connect(db_path)
    query="""with purchase_agg_df as(
    select p.PONumber,
    count(distinct p.Brand) as total_brands,
    sum(p.Quantity) as total_item_quantity,
    sum(p.Dollars)as total_item_dollars,
    avg(julianday(p.ReceivingDate)-julianday(p.PODate)) as avg_receiving_delay
    from purchases p
    group by p.PONumber
    )
    select vi.PONumber,
    vi.Quantity as invoice_quantity,
    vi.Dollars as invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate)-julianday(vi.PODate)) as days_po_to_invoice,
    (julianday(vi.PayDate)-julianday(vi.InvoiceDate)) as days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
    from vendor_invoice vi
    left join purchase_agg_df pa on vi.PONumber=pa.PONumber"""
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    if(abs(row["invoice_dollars"]-row["total_item_dollars"])>5):
        return 1

    if row['avg_receiving_delay']>10:
        return 1

    return 0

def apply_lable(df):
    df['flag_invoice']=df.apply(create_invoice_risk_label,axis=1)
    return df



def data_split(df,features,target):
    X=df[features]
    Y=df[target]
    return train_test_split(X,Y,random_state=42,test_size=0.2)
    
