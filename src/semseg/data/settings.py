from os.path import join
import os 

# data_path = '/opt/data/'
# data_path = 'opt/data/'
data_path = 'opt' + os.sep + 'data' + os.sep
datasets_path = join(data_path, 'datasets')
results_path = join(data_path, 'results')
s3_bucket_name = 'otid-data'