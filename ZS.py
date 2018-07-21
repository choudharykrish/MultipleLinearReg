import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


################## PART-0: AUXILLARY FUNCTIONS #########################

'''
datetime.date(2013, 1, 1).isocalendar()[1]

d = "2013-W0"
r = datetime.datetime.strptime(d + '-0', "%Y-W%W-%w")
print(str(r))

def num_days_month(m):
    if m==1 or m==3 or m==5 or m==7 or m==8 or m==10 or m==12:
        return 31
    if m==4 or m==6 or m==9 or m==11:
        return 30
    if m==2:
        return 28
'''

def startDate(year, week):
    d = str(year) + '-W' + str(week)
    r = datetime.datetime.strptime(d + '-0', "%Y-W%W-%w")
    r = str(r)
    start_date = int(r.split('-')[2].split(' ')[0])
    month = int(r.split('-')[1])
    yr = int(r.split('-')[0])
    return start_date, month,yr
    
# Number of days in each week by month and year
daysByWeek = []
for y in range(2013,2017):
    for week in range(52):
        d_prev_week = -100
        d_week = -100
        ytemp = -100
        wtemp = -100
        mtemp = -100
        start_date,month,year = startDate(y,week)
        if start_date<=7:
            d_prev_week = start_date-1        
        if(week==0):
            wtemp = 53
            mtemp = 12
            ytemp = year-1
        else:
            wtemp = week - 1
            mtemp = month
            ytemp = year
        
        if(startDate(y,week)[0]>startDate(ytemp,wtemp)[0]):
            if(d_week==-100):
                d_week = 7
        else:
            d_prev_week = 7 - startDate(y,week)[0] + 1
            d_week = startDate(y,week)[0] - 1
        temp1 = [y,month, week+1, d_week]
            #daysByWeek.append(temp)
        if(d_prev_week!=-100):
            #if wtemp != 53:
                #wtemp += 1
            if(month==1):
                month = 12
                ytemp = y-1
                week = 52
            else:
                month -= 1
                ytemp = y
            temp = [ytemp,month, week+1, d_prev_week]
            daysByWeek.append(temp)
        if(d_week!=0):
            daysByWeek.append(temp1)
    

def unique(list,index):
    unique_list = []
    for row in list:
        if row[index] not in unique_list:
            unique_list.append(row[index])
    return unique_list



############## PART-1: PREPROCESSING #################################
    


#           HOLIDAYS                #
    
holidays_csv = pd.read_csv('holidays.csv', encoding = "ISO-8859-1")
dataset = holidays_csv.iloc[:406,:3].values

def getHolidaysByCountry(country):
    holidaysByWeek = []
    for holiday in dataset:
        if(str(holiday[1])==country):
            h = []
            temp = str(holiday[0]).split(',')
            week_no=(datetime.date(int((temp[0])), int(temp[1]), int(temp[2])).isocalendar()[1])
            h_name = holiday[2]
            h.append(temp[0])
            h.append(week_no)
            h.append(h_name)
            holidaysByWeek.append(h)
    return np.array(holidaysByWeek)

holiday = dataset[0]
print(holiday)
A_holidays = getHolidaysByCountry('Argentina')
B_holidays = getHolidaysByCountry('Belgium')
C_holidays = getHolidaysByCountry('Columbia')
D_holidays = getHolidaysByCountry('Denmark')
E_holidays = getHolidaysByCountry('England')
F_holidays = getHolidaysByCountry('Finland')


########### ARGENTINA ################

LE_A_holidays = LabelEncoder()
A_holidays[:,2] = LE_A_holidays.fit_transform(A_holidays[:,2])
A_oneHotEncoder = OneHotEncoder(categorical_features=[2])
A_holidays = A_oneHotEncoder.fit_transform(A_holidays).toarray()

print(len(A_holidays[0]))

AHolByWeek = []
unique_A_2 = unique(A_holidays,-2)
for i in unique(A_holidays,-1):
    for year in unique(A_holidays,-2):
        temp = [0 for _ in range(len(A_holidays[0])-2)]
        add = 0
        for A_h in A_holidays:
            if(A_h[-1]==i and A_h[-2]==year):
                add = 1
                for j in range(len(A_holidays[0])-2):
                    temp[j] += A_h[j]
        if(add==1):
            temp.append(year)
            temp.append(i)
            AHolByWeek.append(temp)
            


#               TRAIN               #
            
train_csv = pd.read_csv('yds_train2018.csv')
A_train = []
B_train = []
C_train = []
D_train = []
E_train = []
F_train = []















