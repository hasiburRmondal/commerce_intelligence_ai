def build_discount_analysis(orders):
    total=float(orders['discount_total'].sum()) if 'discount_total' in orders else 0
    discounted=int((orders.get('discount_total',0)>0).sum()) if len(orders) else 0
    return {'discount_total':total,'discounted_orders':discounted,'discount_rate':discounted/len(orders)*100 if len(orders) else 0,'average_discount':total/discounted if discounted else 0}
