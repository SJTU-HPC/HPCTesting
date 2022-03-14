import pandas as pd
import os,time
import glob
import cufflinks as cf
import plotly.offline
import sys
cf.go_offline()
cf.set_config_file(offline=True, world_readable=True)
import torchvision.models as models


MODEL_LIST = {
    # 'mnasnet':models.mnasnet.__all__[1:],
    'resnet': models.resnet.__all__[1:],
    # 'densenet': models.densenet.__all__[1:],
    # 'squeezenet': models.squeezenet.__all__[1:],
    # 'vgg': models.vgg.__all__[1:],
    # 'mobilenet':models.mobilenet.__all__[1:],
    # 'shufflenetv2':models.shufflenetv2.__all__[1:]
}
folder_name='v100/'


csv_list=glob.glob(folder_name+'/*.csv')
columes=[]
for key,values in MODEL_LIST.items():
    for i in values:
        columes.append((key,i))

# print(columes)

for csv in csv_list:
    df=pd.read_csv(csv)
    df.columns = pd.MultiIndex.from_tuples(columes)
    title=csv.split('/')[1].split('_benchmark')[0]
    print(title)
    df1 = df.groupby(level=0,axis=1).mean().mean()
    print(df1)
