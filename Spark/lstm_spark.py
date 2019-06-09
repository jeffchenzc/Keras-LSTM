from pyspark import SparkContext
import numpy as np

sc = SparkContext(appName='LSTMCleaner')

eng_file = "/usr/local/Cellar/apache-spark/2.4.3/data/group/daily_dataset.csv"
wea_file = "/usr/local/Cellar/apache-spark/2.4.3/data/group/weather_daily_darksky.csv"
def handler(row, cat='wea'):
    row = row.rstrip()
    row = row.split(",")
    key = row[1].split()[0]
    if cat == 'eng':
        row = [row[7]]
    else:
        row = [row[0],row[2],row[6],row[7],row[8],row[13],row[20],row[21],row[22]]
    try:
    	row = list(np.array(row).astype(float))
    except:
    	row = None
    return (key, row)


eng = sc.textFile(eng_file)
#eng_header = eng.first()
eng_final = eng.map(lambda row: handler(row, 'eng')).filter(lambda tup: tup[1])\
	.reduceByKey(lambda x, y: [x[0] + y[0]])
wea = sc.textFile(wea_file)
#wea_header = wea.first()
wea_final = wea.map(lambda row: handler(row)).filter(lambda tup: tup[1])
merge = eng_final.join(wea_final).map(lambda tem: (tem[0], tem[1][0]+tem[1][1]))
merge.coalesce(1).saveAsTextFile("clean")