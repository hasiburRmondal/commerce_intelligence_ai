from flask import Response
def dataframe_csv_response(df,filename):
    return Response(df.to_csv(index=False),mimetype='text/csv',headers={'Content-Disposition':f'attachment; filename={filename}'})
