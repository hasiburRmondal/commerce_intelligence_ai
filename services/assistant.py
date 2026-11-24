from services.sales_analysis import build_sales_analysis
from services.product_analysis import build_product_analysis
def answer_question(question,data):
    q=question.lower(); sales=build_sales_analysis(data['orders'],data['items']); products=build_product_analysis(data['items'])
    if any(k in q for k in ['revenue','sales','turnover']): return f"Recorded revenue across included orders is {sales['revenue']:,.2f}. The dataset contains {sales['orders']:,} included orders."
    if any(k in q for k in ['average order','aov']): return f"Average order value is {sales['aov']:,.2f}, calculated as included order revenue divided by unique included orders."
    if any(k in q for k in ['top product','best product','best selling']):
        if products['table'].empty: return 'No product line-item columns with usable quantities were detected.'
        r=products['table'].iloc[0]; return f"Top product by recorded line-item revenue is {r['product_name']} ({r['sku']}), with {r['units']:,.2f} units and {r['revenue']:,.2f} revenue."
    if 'order' in q: return f"There are {sales['orders']:,} included unique orders in the uploaded dataset."
    return 'I can answer questions about total sales/revenue, order count, average order value, and top-selling products. Ask one of those questions.'
