import os, csv, numpy as np

def make_dir(dir):
    if not os.path.exists(dir):
        print("dir ( {} ) is made under {}".format(dir, os.getcwd()))
        os.mkdir(dir)

list2str = lambda s: ', '.join(map(str, s))

def write_csv(rowTitle, row, file_name='res'):
    print('csv Title {}\n row {}'.format(rowTitle, row))
    make_dir('result')
    with open('res/{}.csv'.format(file_name), 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(rowTitle)
        writer.writerow(row)
    csvFile.close()

intersection = lambda lst1, lst2: [value for value in lst1 if value in lst2]


from datetime import datetime
from pytz import timezone, utc
def get_pst_time():
    date_format='%m-%d-%Y--%H-%M-%S-%Z'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    # pstDateTime = pd.Timestamp.today()
    return pstDateTime

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))