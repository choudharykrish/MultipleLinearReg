import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


sales_per_day_col_no = 2
countries = ['England','Denmark','Columbia','Belgium','Argentina','Finland']
################## PART-0: AUXILLARY FUNCTIONS #########################

def num_days_month(m,y):
    if m==1 or m==3 or m==5 or m==7 or m==8 or m==10 or m==12:
        return 31
    if m==4 or m==6 or m==9 or m==11:
        return 30
    if m==2 and y%4==0:
        return 29
    else:
        return 28


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
for y in range(2013,2018):
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
        if(d_prev_week!=-100):
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
A_holidays = getHolidaysByCountry(countries[0])
B_holidays = getHolidaysByCountry(countries[1])
C_holidays = getHolidaysByCountry(countries[2])
D_holidays = getHolidaysByCountry(countries[3])
E_holidays = getHolidaysByCountry(countries[4])
F_holidays = getHolidaysByCountry(countries[5])


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
            

AHolByTime = []
for i in range(1,len(daysByWeek)):
    for row in AHolByWeek:
        if((daysByWeek[i][0]==row[-2]) and (daysByWeek[i][2]==row[-1])):
            temp = [i]
            for j in range(len(row)-2):
                temp.append(row[j])
            AHolByTime.append(temp)


#               TRAIN               #
            
train_csv = pd.read_csv('yds_train2018.csv')
A_train = []
B_train = []
C_train = []
D_train = []
E_train = []
F_train = []

train_df = train_csv.iloc[:,:].values

for row in train_df:
    if row[4]==countries[0]:
        A_train.append(row)
    elif row[4]==countries[1]:
        B_train.append(row)
    elif row[4]==countries[2]:
        C_train.append(row)
    elif row[4]==countries[3]:
        D_train.append(row)
    elif row[4]==countries[4]:
        E_train.append(row)
    elif row[4]==countries[5]:
        F_train.append(row)

print(A_train[0])
        


# MAPPING YEAR and WEEK NUMBER TO TIMELINE INDECES #

A_timeline = []
#[timestamp,  product ID, sales, number of days in that timestamp]
for i in range(1,len(daysByWeek)):
    for row in A_train:
        if((daysByWeek[i][0]==row[0]) and (daysByWeek[i][2]==row[2])):
            temp = [i,row[3],row[5],daysByWeek[i][3]]
            A_timeline.append(temp)


A_compact_timeline = []
# [timestamp, product ID, Total Sales, Number of Days]
for row in A_timeline:
    added = 0
    for i in range(len(A_compact_timeline)):
        if(row[0]==A_compact_timeline[i][0] and row[1]==A_compact_timeline[i][1]):
            A_compact_timeline[i][2] += int(row[2])
            added = 1
    if(added==0):
        A_compact_timeline.append(row)
            

############### PROMOTIONS  ################################

promotions_csv = pd.read_csv('promotional_expense.csv')

promo_df = promotions_csv.iloc[:,:5].values

def Time2Date(timestamp):
    return daysByWeek[timestamp][0], daysByWeek[timestamp][1], daysByWeek[timestamp][2]

def Date2Time(year, month):
    list=[]
    for i in range(1,len(daysByWeek)):
        if daysByWeek[i][0]==year and daysByWeek[i][1]==month:
            list.append(i)
    return list

promoByTime = []
for row in promo_df:
    if row[2] == countries[0]:
        month = row[1]
        year = row[0]
        days = num_days_month(month,year)
        promoPerDay = row[4]/days
        pID = row[3]
        times = Date2Time(year,month)
        for time in times:
            for i in range(len(A_compact_timeline)):
                if A_compact_timeline[i][0] == time and A_compact_timeline[i][1]==row[3]:
                    A_compact_timeline[i].append(promoPerDay)
                    temp = [time,pID, promoPerDay]
                    promoByTime.append(temp)
                    break
    
for row in A_compact_timeline:
    if(len(row)==4):
        row.append(0)

#Getting Sales per day
for i in range(len(A_compact_timeline)):
    A_compact_timeline[i][2] = A_compact_timeline[i][2] / A_compact_timeline[i][3]

for i in range(len(A_compact_timeline)):
    for hol in AHolByTime:
        if(A_compact_timeline[i][0]==hol[0]):
            for j in range(1,len(hol)):
                 A_compact_timeline[i].append(hol[j])
                 
#-----------------------------------------------------------------
n_features = 14
for row in  A_compact_timeline:
    if(len(row)<n_features):
        for i in range(n_features-len(row)):
            row.append(0)
       
### # One Hot Encoding Product ID
A_pID_oneHotEncoder = OneHotEncoder(categorical_features=[1])
A_compact_timeline = A_pID_oneHotEncoder.fit_transform(A_compact_timeline).toarray()
A_compact_timeline = A_compact_timeline[:,1:]

Y = A_compact_timeline[:,sales_per_day_col_no]
X = A_compact_timeline
X = np.delete(X,[sales_per_day_col_no],axis = 1)
#y = Y.reshape(-1,1)

#Squaring sales per day
#for i in range(505):
    #X[i][4] = X[i][4]**1


'''
#splitting the dataset in training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
scaled = sc_x.fit_transform(X_train[:,[3,4]])
X_train = np.delete(X_train,[3,4],axis=1)
X_train = np.concatenate([X_train,scaled],axis=1)

scaled1 = sc_x.transform(X_test[:,[3,4]])
X_test = np.delete(X_test,[3,4],axis=1)
X_test = np.concatenate([X_test,scaled1],axis=1)


sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train.reshape(-1,1))
Y_test = sc_y.transform(Y_test.reshape(-1,1))

'''
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(max_depth = 8)
reg.fit(X,Y)
print('Train Accuracy: ')
print(reg.score(X,Y))
print('Test Accuracy: ')
print(reg.score(X,Y))

#y_pred = reg.predict(X_test)


'''
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
print('Train Accuracy: ')
print(reg.score(X_train,Y_train))
print('Test Accuracy: ')
print(reg.score(X_test,Y_test))



from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_



# Fitting SVR to the dataset
from sklearn.svm import SVR
reg = SVR(kernel = 'linear')
reg.fit(X_train, Y_train)
print('Train Accuracy: ')
print(reg.score(X_train,Y_train))
print('Test Accuracy: ')
print(reg.score(X_test,Y_test))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = reg, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = reg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

'''
'''
#splitting the dataset in training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor


for i in range(1,15):
    print('-------------------------------')
    print(i)
    reg = RandomForestRegressor(max_depth = i)
    reg.fit(X_train,Y_train)
    print('Train Accuracy: ')
    print(reg.score(X_train,Y_train))
    print('Test Accuracy: ')
    print(reg.score(X_test,Y_test))

'''
'''
'''

'''
import numpy as np
np.savetxt("A_compact_timeline.csv", A_compact_timeline, delimiter=",", fmt='%s')
'''





#####################################################################################
###################PREDICTING RESULTS################################################
#####################################################################################

test_csv = pd.read_csv('yds_test2018.csv')

test_df = test_csv.iloc[:,1:5]


testByWeeks  = []
test_df = test_csv.values[:,1:5]
for row in test_df:
    if(row[3] == countries[0]):
        
        temp = []
        date = Date2Time(row[0],row[1])
        for i in date:
            temp = row
            temp = temp.tolist()
            temp.append(i)
            testByWeeks.append(temp)
        
a_test_timeline = []
for row in testByWeeks:
    a_test_timeline.append([row[4],row[2]])
    
for row in a_test_timeline:
    timestamp = row[0]
    days = daysByWeek[timestamp][3]
    row.append(days)



promoByTime = []
for row in promo_df:
    if row[2] == countries[0]:
        month = row[1]
        year = row[0]
        days = num_days_month(month,year)
        promoPerDay = row[4]/days
        pID = row[3]
        times = Date2Time(year,month)
        for time in times:
            for i in range(len(a_test_timeline)):
                if a_test_timeline[i][0] == time and a_test_timeline[i][1]==row[3]:
                    a_test_timeline[i].append(promoPerDay)
                    temp = [time,pID, promoPerDay]
                    promoByTime.append(temp)
                    break

for i in range(len(a_test_timeline)):
    for hol in AHolByTime:
        if(a_test_timeline[i][0]==hol[0]):
            for j in range(1,len(hol)):
                 a_test_timeline[i].append(hol[j])






for row in a_test_timeline:
    if(len(row)<n_features-1):
        for i in range(n_features-len(row)-1):
            row.append(0)

#a_test_timeline =  np.asarray(a_test_timeline)

a_test_with_pID = a_test_timeline

### # One Hot Encoding Product ID
#A_pID_oneHotEncoder1 = OneHotEncoder(categorical_features=[1])
a_test_timeline = A_pID_oneHotEncoder.transform(a_test_timeline).toarray()
a_test_timeline = a_test_timeline[:,1:]



'''
#feature scaling
from sklearn.preprocessing import StandardScaler
scaled_test = sc_x.transform(a_test_timeline[:,[3,4]])
X_train = np.delete(X_train,[3,4],axis=1)
X_train = np.concatenate([X_train,scaled],axis=1)

scaled1 = sc_x.transform(X_test[:,[3,4]])
X_test = np.delete(X_test,[3,4],axis=1)
X_test = np.concatenate([X_test,scaled1],axis=1)
'''


y_pred = reg.predict(a_test_timeline)

for i in range(len(y_pred)):
    y_pred[i] = y_pred[i] * a_test_timeline[i][sales_per_day_col_no]

finaldic = {}    
for j in range(len(daysByWeek)):
    timestamp = a_test_timeline[sales_per_day_col_no]
    for i in range(len(a_test_timeline)):
        if(a_test_timeline[i][sales_per_day_col_no-1] == j):
            key = str(daysByWeek[j][0]) + str(daysByWeek[j][1]) + str(a_test_with_pID[i][1])
            if key in finaldic:
                finaldic[key] += y_pred[i]
            else:
                finaldic[key] = y_pred[i]
                
  
for key in finaldic:
    print (key,       finaldic[key])







