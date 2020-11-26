from datetime import datetime,timedelta
import time
bucket={}
bucket['key_as_string']= '2018-08-06T10:00:00.000Z'

date_ = datetime.strptime(bucket['key_as_string'],"%Y-%m-%dT%H:%M:%S.%fZ")
date = time.mktime(time.strptime(bucket['key_as_string'], '%Y-%m-%dT%H:%M:%S.%fZ'))
print(date)
# date = time.mktime(time.strptime(date, '%Y-%m-%d% %M:%S.%f'))
# print(date_)
#local_time = 2018-08-06 18:00:00

local_time = date_ + timedelta(hours=8)