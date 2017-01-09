import sys
import numpy as np #matrices and data structures
import scipy.stats as ss #standard statistical operations
import pandas as pd #keeps data organized, works well with data
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
import math
import datetime


def linReg(X, y, intercept = False):
    #reweighted least squares logistic regression
    #add intercept:
    if intercept:
        X = np.insert(X, X.shape[1], 1, axis=1)

    y = np.array([y]).T #make column

    #fit regression:
    betas = np.dot(np.dot(np.linalg.inv((np.dot(X.T,X))), X.T), y)

    #calculate p-values:
    error = y - (np.dot(X,betas))
    RSS = np.sum(error**2)
    betas = betas.flatten()
    df = float((X.shape[0] - (len(betas) - 1 if intercept else 0)) - 1)
    s2 = RSS / df
    #print s2
    beta_ses = np.sqrt(s2 / (np.sum( (X - np.mean(X,0))**2, 0)))
    #print beta_ses
    ts = [betas[j] / beta_ses[j] for j in range(len(betas))]
    pvalues = (1 - ss.t(df).cdf(np.abs(ts))) * 2 #two-tailed

    return betas, pvalues

def calc_prob(data):
    #this function will verify how the probability of sales increse or decrease changes given it is a holiday
    prob_increse_by_25_perc = len(data[data['increase'] > 25]) * 1.0 /len(data)
    prob_decrese_by_25_perc = len(data[data['increase'] < -25]) * 1.0/len(data)
    hol = data[data['IsHoliday'] == 1]
    hol['increase'] = (hol['sales_stdz']-hol['prev_week_sales_stdz']/hol['prev_week_sales_stdz']) * 100
    prob_increse_by_25_perc =  len(hol[hol['increase'] > 25]) * 1.0 /len(hol)
    prob_decrese_by_25_perc =  len(hol[hol['increase'] < -25]) * 1.0/len(hol)

#read files
file = 'train.csv'
if (len(sys.argv) > 1) and (sys.argv[1][-4:].lower() == 'csv'):
    file = sys.argv[1]
print "loading %s" % file
data = pd.read_csv(file,sep=',',low_memory=False)
features = pd.read_csv('features.csv', sep=',', low_memory=False)

#join datasets and process columns
result = pd.merge(data, features, on=['Store','Date','IsHoliday'])
result['offers'] = result['MarkDown1'] + result['MarkDown2'] + result['MarkDown3'] + result['MarkDown4']
result['WeekNo'] = result['Date'].apply(lambda val: datetime.datetime.strptime(val, '%Y-%m-%d').date().isocalendar()[1])
#convert isHoliday
def convertBool(val):
    if(val):
        return 1
    else:
        return 0
result['IsHoliday'] = result['IsHoliday'].apply(lambda val: convertBool(val))

#-------------analysis----------------
# take one dept and analyse data
third_dept = result[result['Dept'] == 3]
d_store = third_dept.groupby("Store")
sixth_store = d_store.get_group(6).reset_index()
sixth_store[['Weekly_Sales','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment']].corr()
#sixth_store.Weekly_Sales.plot()

ninth_store = d_store.get_group(9).reset_index()
ninth_store[['Weekly_Sales','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment']].corr()
#ninth_store.Weekly_Sales.plot()

#-------------------------------------

MSES = []
RSQUARE = []
ERRORMSES = []
ERROR_RSQUARE = []

for i in xrange(100):
    try:
        print "-------------------------------------------------------------------------------"
        print " dept :" , i+1
        print "-----------------"
        dataset = result[result['Dept'] == i+1]
        dataset.reindex()

        dataset = dataset[['Store', 'Date', 'Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'offers','WeekNo']]

        #hypothesis testing
        #see correlation matrix
        dataset[['Weekly_Sales','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','offers']].corr()
        test = dataset[['Weekly_Sales','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment']]
        test = test.dropna()
        y = test['Weekly_Sales']
        X = test[['IsHoliday','Temperature','Fuel_Price','CPI','Unemployment']]

        y = (y - y.mean()) / y.std()
        X = (X - X.mean()) / X.std()
        betas, pvalues = linReg(X, y)
        env = ['IsHoliday',  'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        #print 'betas  p-values:'
        #print zip(env, betas, pvalues)#DEBUG


        #standardize sales value across stores
        def standardize_sales(store):
            store['store_mean'] = store['Weekly_Sales'].mean()
            store['store_std'] = store['Weekly_Sales'].std()
            store['sales_stdz'] = (store['Weekly_Sales'] - store['Weekly_Sales'].mean())/ store['Weekly_Sales'].std()
            return store
        dataset = dataset.groupby('Store').apply(standardize_sales)

        #avg_store_week_sales
        def avg_store_week_sales(row):
            avg = dataset[(dataset['Store'] == row['Store']) & (dataset['WeekNo'] == row['WeekNo']) & (dataset['Date'] < row['Date'])]['sales_stdz'].mean()
            if math.isnan(avg):
                avg = dataset[(dataset['Store'] == row['Store']) & (dataset['WeekNo'] == row['WeekNo'])]['sales_stdz'].mean()
            return avg;
        dataset['avg_store_week_sales'] = dataset.apply(lambda row: avg_store_week_sales(row), axis=1)

        def avg_store_next_week_sales(row):
            avg = dataset[(dataset['Store'] == row['Store']) & (dataset['WeekNo'] == row['WeekNo'] %52 +1) & (dataset['Date'] < row['Date'])]['sales_stdz'].mean()
            if math.isnan(avg):
                avg = dataset[(dataset['Store'] == row['Store']) & (dataset['WeekNo'] == row['WeekNo'] %52 +1)]['sales_stdz'].mean()
            return avg;
        dataset['avg_store_next_week_sales'] = dataset.apply(lambda row: avg_store_next_week_sales(row), axis=1)

        def avg_store_prev_week_sales(row):
            avg = dataset[(dataset['Store'] == row['Store']) & (dataset['WeekNo'] == (row['WeekNo'] - 2)%52 +1) & (dataset['Date'] < row['Date'])]['sales_stdz'].mean()
            if math.isnan(avg):
                avg = dataset[(dataset['Store'] == row['Store']) & (dataset['WeekNo'] == (row['WeekNo'] - 2)%52 +1)]['sales_stdz'].mean()
            return avg;
        dataset['avg_store_prev_week_sales'] = dataset.apply(lambda row: avg_store_prev_week_sales(row), axis=1)

        def prev_week_sales(val):
            prev_sales = []
            prev_sales.append(val['Weekly_Sales'][:1][val['Weekly_Sales'][:1].keys()[0]])

            # For each row in the column,
            for row in val['Weekly_Sales'][:-1]:
                prev_sales.append(row)

            val['prev_week_sales'] = prev_sales
            return val

        dataset = dataset.groupby('Store').apply(prev_week_sales)

        dataset['prev_week_sales_stdz'] = (dataset['prev_week_sales'] - dataset['store_mean'])/ dataset['store_std']

        #effect of sales
        #probability of incresing sales by atleast 25% given week has special holiday
        #calc_prob(dataset)

        def avg_overall_week_sales(row):
            avg = dataset[(dataset['WeekNo'] == row['WeekNo']) & (dataset['Date'] < row['Date'])]['sales_stdz'].mean()
            if math.isnan(avg):
                avg = dataset[(dataset['WeekNo'] == row['WeekNo'])]['sales_stdz'].mean()
            return avg;
        dataset['avg_overall_week_sales'] = dataset.apply(lambda row: avg_overall_week_sales(row), axis=1)


        def avg_overall_next_week_sales(row):
            avg = dataset[(dataset['WeekNo'] == row['WeekNo'] %52 +1) & (dataset['Date'] < row['Date'])]['sales_stdz'].mean()
            if math.isnan(avg):
                avg = dataset[(dataset['WeekNo'] == row['WeekNo'] %52 +1)]['sales_stdz'].mean()
            return avg;
        dataset['avg_overall_next_week_sales'] = dataset.apply(lambda row: avg_overall_next_week_sales(row), axis=1)


        def avg_overall_prev_week_sales(row):
            avg = dataset[(dataset['WeekNo'] == (row['WeekNo'] - 2)%52 +1) & (dataset['Date'] < row['Date'])]['sales_stdz'].mean()
            if math.isnan(avg):
                avg = dataset[(dataset['WeekNo'] == (row['WeekNo'] - 2)%52 +1)]['sales_stdz'].mean()
            return avg;
        dataset['avg_overall_prev_week_sales'] = dataset.apply(lambda row: avg_overall_prev_week_sales(row), axis=1)


        def yieldNFolds(X, y, folds = 10, dev = False):
            n = len(y) / folds
            extra = len(y) % folds
            i = 0
            for f in xrange(folds):
                j = i+n
                if extra > 0:
                    j+=1
                    extra -= 1

                ytrain = np.concatenate((y[:i], y[j:])).copy()
                ytest = y[i:j].copy()
                Xtrain = np.concatenate((X[:i], X[j:])).copy()
                Xtest = X[i:j].copy()
                i = j
                if dev:#if including a development set:
                    (ydev, ytrain) = (ytrain[:n].copy(), ytrain[n:].copy())
                    (Xdev, Xtrain) = (Xtrain[:n].copy(), Xtrain[n:].copy())
                    yield Xtrain, ytrain, Xtest, ytest, Xdev, ydev
                else:
                    yield Xtrain, ytrain, Xtest, ytest


        dataset[['sales_stdz', 'avg_overall_week_sales','avg_overall_next_week_sales','avg_overall_prev_week_sales','avg_store_week_sales','avg_store_next_week_sales','avg_store_prev_week_sales','prev_week_sales_stdz']].corr()

        #randomly permutate data
        data = dataset.iloc[np.random.permutation(len(dataset))]
        data = data[['sales_stdz','avg_overall_week_sales','avg_overall_next_week_sales','avg_overall_prev_week_sales','avg_store_week_sales','avg_store_next_week_sales','avg_store_prev_week_sales','prev_week_sales_stdz']]
        data = data.dropna()
        y = data['sales_stdz']
        X = data[['avg_overall_week_sales','avg_overall_next_week_sales','avg_overall_prev_week_sales','avg_store_week_sales','avg_store_next_week_sales','avg_store_prev_week_sales','prev_week_sales_stdz']]

        sales_mean = y.mean()
        sales_std = y.std()
        y = (y - y.mean()) / y.std()
        X = (X - X.mean()) / X.std()

        betas, pvalues = linReg(X, y)
        env = ['avg_overall_week_sales','avg_overall_next_week_sales','avg_overall_prev_week_sales','avg_store_week_sales','avg_store_next_week_sales','avg_store_prev_week_sales','prev_week_sales_stdz']
        print 'betas  pvalues :'
        print zip(env, betas, pvalues)#DEBUG


        #fit the model for each fold, and track y_hats, y_trues
        y_hats, y_trues = [], []
        for Xtrain, ytrain, Xtest, ytest in yieldNFolds(X, y, 10):
            model = sm.OLS(ytrain, Xtrain).fit()
            y_hats.extend(model.predict(Xtest))
            y_trues.extend(ytest) #just in case ytest is not in the order of y

        print "Non-regularized Linear Regresssion MSE: %.3f" % np.mean((np.array(y_trues) - np.array(y_hats))**2)
        print ss.pearsonr(y_trues, y_hats)
        MSES.append(np.mean((np.array(y_trues) - np.array(y_hats))**2))

        #Ridge Regression
        y_hats, y_trues = {'ridge':[]}, {'ridge':[]}
        for Xtrain, ytrain, Xtest, ytest, Xdev, ydev in yieldNFolds(X, y, 10, dev=True):
            min_mse = np.inf
            min_model = None
            for alpha in [10**i for i in range(-5, 5)]:
                model = Ridge(alpha=alpha).fit(Xtrain, ytrain)
                mse = np.mean((np.array(ydev) - np.array(model.predict(Xdev)))**2)
                if mse < min_mse:
                    min_mse = mse
                    min_model = model
            #print "min mse %.3f; selected model:" %min_mse, min_model
            ypred = min_model.predict(Xtest)
            y_hats['ridge'].extend(ypred)
            y_trues['ridge'].extend(ytest) #just in case ytest is not in the order of y

        #LASSO Regression
        y_hats['lasso'], y_trues['lasso'] = [], []
        for Xtrain, ytrain, Xtest, ytest, Xdev, ydev in yieldNFolds(X, y, 10, dev=True):
            min_mse = np.inf
            min_model = None
            for alpha in [10**i for i in range(-5, 5)]:
                model = Lasso(alpha=alpha).fit(Xtrain, ytrain)
                mse = np.mean((np.array(ydev) - np.array(model.predict(Xdev)))**2)
                if mse < min_mse:
                    min_mse = mse
                    min_model = model
            #print "min mse %.3f; selected model:" %min_mse, min_model
            ypred = min_model.predict(Xtest)
            y_hats['lasso'].extend(ypred)
            y_trues['lasso'].extend(ytest) #just in case ytest is not in the order of y

        for k in y_hats.keys():
            print "%s Regularized MSE: %.3f" % \
                (k, np.mean((np.array(y_trues[k]) - np.array(y_hats[k]))**2))
            print ss.pearsonr(y_trues[k], y_hats[k])


        #select best model and fit the data
        model = sm.OLS(y, X).fit()
        print model.summary()
        X['y_hats'] = model.predict(X)
        RSQUARE.append(model.rsquared)


        X['sales_stdz'] = dataset['sales_stdz']
        X['error'] = X['sales_stdz'] - X['y_hats']
        X['Temperature'] = dataset['Temperature']
        X['Fuel_Price'] = dataset['Fuel_Price']
        X['CPI'] = dataset['CPI']
        X['Unemployment'] = dataset['Unemployment']

        # X.corr()


        errory = X['error']
        errorX = X[['Temperature','Fuel_Price', 'CPI','Unemployment']]
        error_mean = y.mean()
        error_std = y.std()
        errory = (errory - errory.mean()) / errory.std()
        errorX = (errorX - errorX.mean()) / errorX.std()

        model = sm.OLS(errory, errorX).fit()
        print "error rsqare  " , model.rsquared
        ERROR_RSQUARE.append(model.rsquared)
        # print model.summary()

        y_hats, y_trues = [], []
        for Xtrain, ytrain, Xtest, ytest in yieldNFolds(errorX, errory, 10):
            model = sm.OLS(ytrain, Xtrain).fit()
            y_hats.extend(model.predict(Xtest))
            y_trues.extend(ytest) #just in case ytest is not in the order of y

        print "Non-regularized Linear Regresssion MSE for errors: %.3f" % np.mean((np.array(y_trues) - np.array(y_hats))**2)
        ERRORMSES.append(model.rsquared)
        errorX['y_hats'] = model.predict(errorX)

        print MSES
        print RSQUARE
        print ERRORMSES
        print ERROR_RSQUARE
        print "-------------------------------------------------------------------------------"

    except:
        print "exception ", i+1

print MSES
print RSQUARE
print ERRORMSES
print ERROR_RSQUARE