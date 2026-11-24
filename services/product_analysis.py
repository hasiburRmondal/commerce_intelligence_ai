import pandas as pd
def build_product_analysis(items):
    if items.empty: return {'table':pd.DataFrame(columns=['product_name','sku','units','revenue','orders']),'top_labels':[],'top_values':[],'count':0}
    t=items.groupby(['product_name','sku'],dropna=False).agg(units=('quantity','sum'),revenue=('line_total','sum'),orders=('order_id','nunique')).reset_index().sort_values('revenue',ascending=False)
    t['revenue']=t['revenue'].round(2); t['units']=t['units'].round(2)
    return {'table':t,'top_labels':t.head(10)['product_name'].astype(str).tolist(),'top_values':[round(float(x),2) for x in t.head(10)['revenue']], 'count':len(t)}
