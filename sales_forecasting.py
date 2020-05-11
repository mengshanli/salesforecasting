# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from scipy import stats
from sklearn import preprocessing

# input raw data
def rawdata_id():
    raw_data=pd.read_csv("model_raw_data.csv", index_col=[0])
    #sales_condition=raw_data.groupby(['id'])['sales'].sum().reset_index(drop=False)
    id102=pd.read_csv('id102.csv').reset_index(drop=True)['id']    
    return raw_data, id102

# sMAPE(Symmetric mean absolute percentage error)
def smape_new(y_true, y_pred):     
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true1, y_pred1=[], []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
           y_true1.append(y_true[i])
           y_pred1.append(y_pred[i])

    y_true1, y_pred1 = np.array(y_true1), np.array(y_pred1)    
    ans=np.sum(abs(y_true1 - y_pred1) / (abs(y_true1) + abs(y_pred1))) /len(y_true) *100
    return ans

# 計算相關性，產出季節性週期長度
def seasonality_period(y):
    y=y.reset_index(drop=True)
    for i in range(100):   # 平移歷史銷售值(移成101欄；若設成105，導致太多NAN)
        if i==0:
            tem=y.shift(periods=1, freq=None)
        else:
            tem=y.iloc[:,i].shift(periods=1, freq=None)    
        y=pd.concat([y, tem], axis=1)
    
    corr_tem=y.dropna().corr() # 計算相關係數    
    corr_tem=corr_tem.iloc[0,:] 
    corr_tem=corr_tem.reset_index(drop=True)        
    corr_tem=corr_tem[corr_tem != 1] # 剔除自己與自己比(相關性=1)
    
    result=corr_tem[:24] #將季節性週期長度控制在24週中
    sea_period=result.idxmax() # 季節性週期長度:在相關係數大於0.7中，週期最小的    
    sea_period_corr=result[sea_period] #季節性週期長度的相關性        
    return sea_period, sea_period_corr

#去季節性計算中的參數計算
def desea_tem(demand, i): #第i期的值
    if i > 105:
        i=105    
    return(demand.get_value(i))

def desea_sigma(start, end, demand):
    if end > 105:
        end=105
    return(2*(sum(desea_tem(demand, i) for i in range(start, end+1))))
     
def desea_sigma_odd(start,end,p, demand):
    if end > 105:
        end=105
    return(sum(desea_tem(demand, i)/p for i in range(start, end+1)))


'''
Holt's model
'''
# Holt's model
def Holt(alpha, beta, demand, level, trend):
    demand=pd.DataFrame(demand)
    levels=[]
    trends=[]
    forecasts=[]
    
    levels.append(level)
    trends.append(trend)
            
    for i in range(1,len(demand)+1):  
        level=alpha*(demand['sales'][i])+(1-alpha)*(levels[i-1]+trends[i-1])
        trend=beta*(level-levels[i-1])+(1-beta)*(trends[i-1])
        forecast=level+trend
    
        levels.append(level)
        trends.append(trend)        
        forecasts.append(forecast)
    
    demand['forecast']=forecasts
        
    # 無條件進位
    demand['forecast']=np.ceil(demand['forecast'])    
    # 負數為0；-0=>0
    demand['forecast'][demand['forecast']< 0] = 0
    demand['forecast'][demand['forecast'] == -0] = 0

    smape_ans=smape_new(demand['sales'], demand['forecast'])      
    return demand, smape_ans

#Holt's model
def range_para_holt(alpha, beta): #[0,1]
    if alpha > 1:
        alpha=1
    elif beta > 1:
        beta=1
    elif alpha < 0:
        alpha=0.01
    elif beta < 0:
        beta=0.01
    return alpha, beta


# Holt's model: adjust beta 
def adjustpara_holt(alpha, beta, demand, level, trend, min_smape, highlow):
    smape_result2=[]
    for i in np.linspace(0.1,1, num=30):
        if highlow == 'high':
            alpha, beta= alpha, beta-i
        else:
            alpha, beta= alpha, beta+i

        alpha, beta=range_para_holt(alpha, beta)
        
        demand, smape_ans=Holt(alpha, beta, demand, level, trend)
        final=alpha, beta, smape_ans
        smape_result2.append(final) 
        
    smape_result2=pd.DataFrame(smape_result2)
    min_index2=smape_result2[2].idxmin()
    min_smape2=smape_result2[2].min()
    
    if min_smape2 < min_smape:
        alpha=smape_result2.iloc[min_index2][0]
        beta=smape_result2.iloc[min_index2][1]        
    return alpha, beta, min_smape2

# Holt's model
def model_holt(data_id, demand):                
    # 產出迴歸式 & deseasonalized demand & seasonal factor    
    x=np.linspace(1, len(data_id)+1, num=len(data_id), endpoint=False)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data_id['sales'])
            
    # level 和 trend的初始值 (initial value)    
    level=intercept  
    trend=slope 
  
    smape_result1=[]
    for i in np.linspace(0.1, 1, num=30):
        alpha, beta= i, i
        alpha, beta=range_para_holt(alpha, beta)
        
        demand, smape_ans=Holt(alpha, beta, demand, level, trend)
        final=alpha, beta, smape_ans
        smape_result1.append(final) 
            
    smape_result1=pd.DataFrame(smape_result1)
    min_index=smape_result1[2].idxmin()
    min_smape=smape_result1[2].min()
    
    alpha=smape_result1.iloc[min_index][0]
    beta=smape_result1.iloc[min_index][1]
        
    if alpha >= 0.5:
        alpha, beta, finalmin_mape=adjustpara_holt(alpha, beta, demand, level, trend, min_smape, 'high')
    else:
        alpha, beta, finalmin_mape=adjustpara_holt(alpha, beta, demand, level, trend, min_smape,  'low')
    
    # 以最優化參數預測    
    demand, smape_holt=Holt(alpha, beta, demand, level, trend)     
    demand=demand.reset_index(drop=True)
    demand_holt=demand.copy()    
    return demand_holt, smape_holt

''''
Winter's model
'''
#將Winter's model的參數控制在[0,1]
def range_para(alpha, beta, gamma): 
    if alpha > 1:
        alpha=1
    elif beta > 1:
        beta=1
    elif gamma > 1:
        gamma=1
    elif alpha < 0:
        alpha=0.01
    elif beta < 0:
        beta=0.01
    elif gamma < 0:
        gamma=0.01
    return alpha, beta, gamma

# Winter's model: adjust beta & gamma
def adjustpara(alpha, beta, gamma, demand, level, trend, seasonalfac, min_smape, highlow):

    smape_result2=[]
    for i in np.linspace(0.1,1, num=30):
        if highlow == 'high':
            alpha, beta, gamma= alpha, beta-i, gamma  
        else:
            alpha, beta, gamma= alpha, beta+i, gamma  

        alpha, beta, gamma=range_para(alpha, beta, gamma)
        
        demand, smape_ans=HW(alpha, beta, gamma, demand, level, trend, seasonalfac)
        final=alpha, beta, gamma, smape_ans
        smape_result2.append(final) 
        
    smape_result2=pd.DataFrame(smape_result2)
    min_index2=smape_result2[3].idxmin()
    min_smape2=smape_result2[3].min()
    
    if min_smape2 < min_smape:
        alpha=smape_result2.iloc[min_index2][0]
        beta=smape_result2.iloc[min_index2][1]
        gamma=smape_result2.iloc[min_index2][2]
    
    # adjust gamma
    smape_result3=[]
    for i in np.linspace(0.1,1, num=30):
        if highlow == 'high':
            alpha, beta, gamma= alpha, beta, gamma-i
        else:
            alpha, beta, gamma= alpha, beta, gamma+i
            
        alpha, beta, gamma=range_para(alpha, beta, gamma)
        
        demand, smape_ans=HW(alpha, beta, gamma, demand, level, trend, seasonalfac)
        final=alpha, beta, gamma, smape_ans
        smape_result3.append(final) 
       
    smape_result3=pd.DataFrame(smape_result3)
    min_index3=smape_result3[3].idxmin()
    min_smape3=smape_result3[3].min()
    
    if min_smape3 < min_smape2:
        alpha=smape_result3.iloc[min_index3][0]
        beta=smape_result3.iloc[min_index3][1]
        gamma=smape_result3.iloc[min_index3][2]
        min_smape2=min_smape3
    
    return alpha, beta, gamma, min_smape2

# Winter's model 預測銷售量
def HW(alpha, beta, gamma, demand, level, trend, seasonalfac):

    levels, trends, forecasts, seafactors=[], [], [], seasonalfac    
    levels.append(level)
    trends.append(trend)
            
    for i in range(1,len(demand)+1): 
        level=alpha*(demand['sales'][i]/seafactors[i-1])+(1-alpha)*(levels[i-1]+trends[i-1])
        trend=beta*(level-levels[i-1])+(1-beta)*(trends[i-1])
        seafac=gamma*(demand['sales'][i]/level)+(1-gamma)*seafactors[i-1]
        forecast=level+trend
    
        levels.append(level)
        trends.append(trend)
        seafactors.append(seafac)       
        forecasts.append(forecast)
    
    seafactors1=seafactors[:len(demand)]
    demand['forecast_desea']=forecasts
    demand['seafactor']=seafactors1
    demand['forecast']=demand['forecast_desea']*demand['seafactor']
    
    # 無條件進位
    demand['forecast']=np.ceil(demand['forecast'])    
    # 負數為0；-0=>0
    demand['forecast'][demand['forecast']< 0] = 0
    demand['forecast'][demand['forecast'] == -0] = 0
    
    smape_ans=smape_new(demand['sales'], demand['forecast'])     
    return demand, smape_ans

# Winter's model
def model_winter(data_id, demand):
        
    sea_period, sea_period_corr=seasonality_period(data_id['sales']) # 季節週期長度
       
    p=sea_period
    p_start=p-1 # 預測去季節化需求的第一期 (t)
    results=[] 
    for t in range(p_start, len(demand)-1): 
        if p % 2 == 0: # p is even (偶數)   
            desea_demand_tem=(desea_tem(demand, t-(p/2))+desea_tem(demand, t+(p/2))+desea_sigma(int(t+1-(p/2)),int(t-1+(p/2)), demand))/(2*p)
        else: # p is odd (奇數)  
            desea_demand_tem=desea_sigma_odd(int(t-((p-1)/2)),int(t+((p-1)/2)), p, demand)
        results.append(desea_demand_tem)
        
    # 產出迴歸式 & deseasonalized demand & seasonal factor    
    x=np.linspace(1, len(results)+1,num=len(results), endpoint=False)
    y=results
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    #計算去季節性因素的銷售量
    tem=[]
    for i in range(1, len(demand)+1):
        tem.append(intercept+slope*i) #套入迴歸式      
    tem=pd.DataFrame(tem, columns=['desea_demand']) # 去季節性因素的銷售量(deseasonalized demand)
    tem.index += 1
    
    demand=pd.concat([demand,tem], axis=1)
    demand['sea_factor']=demand['sales']/demand['desea_demand'] # 計算季節性因素(seasonal factor)
    
    # 計算每一個季節週期平均季節性因素值
    seasonal_factors=[]
    for start in range(1, p+1):
        tem=sum(demand['sea_factor'].get_value(i) for i in range(start,len(demand)+1,p))
        seasonal_factors.append(tem/(len(demand) // p)) 

    # 初始值(initial value)
    level=intercept  
    trend=slope 
    seasonalfac=seasonal_factors.copy()  
    
    # 找出 使sMAPE最低的alpha, beta, gamma的組合，只截取alpha
    smape_result1=[]
    for i in np.linspace(0.1,1, num=30):
        alpha, beta, gamma= i, i, i
        alpha, beta, gamma=range_para(alpha, beta, gamma)
        
        demand, smape_ans=HW(alpha, beta, gamma, demand, level, trend, seasonalfac)
        final=alpha, beta, gamma, smape_ans
        smape_result1.append(final) 
            
    smape_result1=pd.DataFrame(smape_result1)
    min_index=smape_result1[3].idxmin()
    min_smape=smape_result1[3].min()
    
    alpha=smape_result1.iloc[min_index][0]
    beta=smape_result1.iloc[min_index][1]
    gamma=smape_result1.iloc[min_index][2]
    
    # alpha已固定，接著調整beta & gamma  
    if alpha >= 0.5:
        alpha, beta, gamma, finalmin_smape=adjustpara(alpha, beta, gamma, demand, level, trend, seasonalfac, min_smape, 'high')
    else:
        alpha, beta, gamma, finalmin_smape=adjustpara(alpha, beta, gamma, demand, level, trend, seasonalfac, min_smape,  'low')
    
    # 以最優化參數預測     
    demand, smape_winter=HW(alpha, beta, gamma, demand, level, trend, seasonal_factors)     
    demand=demand.reset_index(drop=True)
    demand_winter=demand.copy()
    
    return demand_winter, smape_winter

'''
XGBoost 
'''
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
#from xgboost import plot_importance

def xgboost(x1, y):               
    features_all=all_features()
    
    no=-1 
    smape_result_xgb=[]    
    for no in range(-1, len(features_all)):
        # 透過剔除特徵的方式篩選較佳特徵組合
        if no == -1:
            x1_drop=x1     
        else:
            x1_drop=x1.drop(features_all[no], axis=1)
        
        # 以前80%(Train)歷史銷售量預測後20%(Test)銷售量     
        X_train, X_test, y_train, y_test = x1_drop[:84], x1_drop[84:], y[:84], y[84:]
                
        xgb1 = XGBRegressor()       
        my_cv=TimeSeriesSplit(n_splits=10) 
        
        parameters_xgb = {'learning_rate': [0.01, 0.05, 0.1, 0.2],  
                      'max_depth': [3,5,7, 10],
                      'n_estimators': [550, 575, 600]}
        #交叉驗證 
        xgb_grid = GridSearchCV(xgb1,
                                parameters_xgb,
                                scoring='neg_mean_squared_error', 
                                cv = my_cv,
                                n_jobs = -1,
                                verbose=True)
        
        xgb_grid.fit(X_train, y_train) 
        
        # 找出最佳的learning rate 和 max depth
        best_para=xgb_grid.best_params_
        lr, depth, n_esti=best_para['learning_rate'], best_para['max_depth'], best_para['n_estimators']
        
        # testing data
        predict_test= xgb_grid.predict(X_test)
        predict_test=predict_test.round(0) # 四捨五入
        predict_test[predict_test == -0]=0  #負0為0          
        smape_xgb_test=smape_new(y_test, predict_test)

        # training data
        predict_train= xgb_grid.predict(X_train)
        predict_train=predict_train.round(0) # 四捨五入
        predict_train[predict_train == -0]=0  #負0為0           
        smape_xgb_train=smape_new(y_train, predict_train)

        # all data
        predict_all= xgb_grid.predict(x1_drop)
        predict_all=predict_all.round(0) # 四捨五入
        predict_all[predict_all == -0]=0 #負0為0             
        smape_xgb_all=smape_new(y, predict_all)

        if no == -1: # 如果最佳模型為"無剔除任一特徵"
            smape_result='no', lr, depth, n_esti, smape_xgb_test, smape_xgb_train, smape_xgb_all
        else:    
            smape_result=features_all[no], lr, depth, n_esti, smape_xgb_test, smape_xgb_train, smape_xgb_all
            
        smape_result_xgb.append(smape_result)
   
    smape_result_xgb_df=pd.DataFrame(smape_result_xgb, columns=['drop_feature', 'learning_rate', 'max_depth', 'n_estimators',
                                                                'smape_test', 'smape_train', 'smape_all'])
    
    min_test=smape_result_xgb_df['smape_test'].min()  
    min_index=smape_result_xgb_df['smape_test'].where(smape_result_xgb_df['smape_test'] ==min_test).first_valid_index()
    
    
    feature_drop=smape_result_xgb_df.iloc[min_index,0] # 測試與訓練績效差異最小時，是丟棄了哪些特徵
    min_lr=smape_result_xgb_df.iloc[min_index,1] # 測試與訓練績效差異最小時的learning rate
    min_depth=smape_result_xgb_df.iloc[min_index,2] # 測試與訓練績效差異最小時的max_depth
    min_n_esti=smape_result_xgb_df.iloc[min_index,3] # 測試與訓練績效差異最小時n_estimators
    final_test= smape_result_xgb_df.iloc[min_index,4]
    final_train= smape_result_xgb_df.iloc[min_index,5]
    final_all= smape_result_xgb_df.iloc[min_index,6]
    
    '''
    # 特徵重要性分數(權重)          
    X_train, X_test, y_train, y_test = x1[:84], x1[84:], y[:84], y[84:]
    
    xgb_plot=XGBRegressor(max_depth=min_depth, learning_rate=min_lr, n_estimators= min_n_esti)           
    xgb_plot.fit(X_train, y_train)
        
    smape_per="{:.2f}%".format(final_test)
    plot_importance(xgb_plot)
    plt.title('Feature Importance - %s (sMAPE= %s)' % (item_id, smape_per))
    plt.savefig('%s_Feature Importance.png' % item_id, dpi=300, bbox_inches='tight')
    plt.show()
    '''    
    # 以最佳特徵組合進行預測 (presict all with the best model)
    if feature_drop == 'no':
        x1_drop=x1    
    else:
        x1_drop=x1.drop(feature_drop, axis=1)
            
    X_train, X_test, y_train, y_test = x1_drop[:84], x1_drop[84:], y[:84], y[84:]

    xgb_forecast=XGBRegressor(max_depth=min_depth, learning_rate=min_lr, n_estimators= min_n_esti)           
    xgb_forecast.fit(X_train, y_train)
    
    predict_all= xgb_forecast.predict(x1_drop)
    predict_all=predict_all.round(0) # 四捨五入
    predict_all[predict_all == -0]=0  #負0為0
        
    return  feature_drop, final_test, final_train, final_all, predict_all

'''
SVR
參數: C越高、epsilon越低=> 不容忍錯誤 => 容易 overfitting
'''
from sklearn.svm import SVR
   
def svr(x1, y):  
    features_all=all_features()
    
    #剔除特徵組合以篩選出最佳特徵組合
    smape_result=[]
    for no in range(-1, len(features_all)): 
        if no ==-1:
            x1_drop=x1
        else:
            x1_drop=x1.drop(features_all[no], axis=1)
        
        X_train, X_test, y_train, y_test = x1_drop[:84], x1_drop[84:], y[:84], y[84:]
        my_cv1 = TimeSeriesSplit(n_splits=10)
        
        svr1 = SVR(kernel='rbf', gamma='auto') 
        
        parameters_svr = {'C': [0.5,2,4,6,10],  #cost
                      'epsilon': [0,0.1,0.5,0.75,1]} 
        
        #交叉驗證        
        svr_grid = GridSearchCV(svr1,
                                parameters_svr,
                                scoring='neg_mean_squared_error', 
                                cv = my_cv1,
                                n_jobs = -1,
                                verbose=True)
        
        svr_grid.fit(X_train, y_train) 
        
        best_para=svr_grid.best_params_
        c, epsilon=best_para['C'], best_para['epsilon']
       
        # testing data
        predict_test= svr_grid.predict(X_test)
        predict_test=predict_test.round(0) # 四捨五入
        predict_test[predict_test == -0]=0 #負0為0       
        smape_svr_test=smape_new(y_test, predict_test)

        # training data
        predict_train= svr_grid.predict(X_train)
        predict_train=predict_train.round(0) # 四捨五入
        predict_train[predict_train == -0]=0 #負0為0              
        smape_svr_train=smape_new(y_train, predict_train)

        # all data
        predict_all= svr_grid.predict(x1_drop)
        predict_all=predict_all.round(0) # 四捨五入
        predict_all[predict_all == -0]=0 #負0為0            
        smape_svr_all=smape_new(y, predict_all)
                                
        if no == -1:
            anss='no', c, epsilon, smape_svr_test, smape_svr_train, smape_svr_all
        else:
            anss=features_all[no], c, epsilon, smape_svr_test, smape_svr_train, smape_svr_all
        smape_result.append(anss)
        
    smape_result_svr_df=pd.DataFrame(smape_result, columns=['drop_feature', 'C', 'epsilon', 'smape_test', 'smape_train', 'smape_all'])
        
    min_test=smape_result_svr_df['smape_test'].min()  
    min_index=smape_result_svr_df['smape_test'].where(smape_result_svr_df['smape_test'] ==min_test).first_valid_index()
    
    feature_drop=smape_result_svr_df.iloc[min_index,0] # 測試與訓練績效差異最小時，是丟棄了哪些特徵
    final_test= smape_result_svr_df.iloc[min_index,3]
    final_train= smape_result_svr_df.iloc[min_index,4]
    final_all= smape_result_svr_df.iloc[min_index,5]
    min_c=smape_result_svr_df.iloc[min_index,1]
    min_epsilon=smape_result_svr_df.iloc[min_index,2]
    
    # 以最佳特徵組合進行預測 (presict all with the best model)
    if feature_drop == 'no':
        x1_drop=x1     
    else:
        x1_drop=x1.drop(feature_drop, axis=1)   
    
    X_train, X_test, y_train, y_test = x1_drop[:84], x1_drop[84:], y[:84], y[84:]

    my_cv1 = TimeSeriesSplit(n_splits=10)        
    svr1 = SVR(kernel='rbf', gamma='auto',C=min_c, epsilon=min_epsilon) #gamma='auto'        
    svr1.fit(X_train, y_train) 
    
    predict_all= svr1.predict(x1_drop)
    predict_all=predict_all.round(0) # 四捨五入
    predict_all[predict_all == -0]=0 #負0為0 
                
    return  feature_drop, final_test, final_train, final_all, predict_all

# 所有特徵的排列組合
def all_features():

    promotions=['event_0','event_1', 'event_2', 'event_3','event_4', 'event_5', 'event_6']
    #features=['employee_nums', 'holiday_nums', 'price', promotions]
        
    features_all=[['employee_nums'],
    ['employee_nums', 'holiday_nums'],
    ['employee_nums', 'holiday_nums', 'price'],
    ['employee_nums', 'holiday_nums', ['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']],
    ['employee_nums', 'price'],
    ['employee_nums', 'price', ['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']],
    ['employee_nums', ['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']],
    ['holiday_nums'],
    ['holiday_nums', 'price'],
    ['holiday_nums', 'price', ['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']],
    ['holiday_nums', ['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']],
    ['price'],
    ['price', ['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']],
    [['event_0', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5', 'event_6']]]
    
    for i in range(len(features_all)): #將list提出來
        for j in range(len(features_all[i])):        
            if features_all[i][j] == promotions:
                del features_all[i][j]
                for p in promotions:
                    features_all[i].append(p)
    
    return features_all

# XGBoost & SVR
def machine_learning(data_id, demand):        
    x=data_id.drop(['id','week','sales'], axis=1)
    y=data_id['sales'].reset_index(drop=True)
    
    #normalization
    x1=pd.DataFrame(preprocessing.normalize(x, norm='l2'))
    x1.columns=x.columns
    x1['exp_forecast']=demand['forecast'].reset_index(drop=True)
        
    xgboost_drop, xgboost_test, xgboost_train, xgboost_all, xgboost_predict_all=xgboost(x1, y)    
    svr_drop, svr_test, svr_train, svr_all, svr_predict_all=svr(x1, y)
    
    # 比較 'xgboost' & 'svr' 訓練集精準度，取sMAPE較低的模型
    if xgboost_train >= svr_train:
        model_ml, all_ml ='svr', svr_train
        feature_drop=svr_drop
        demand_ml=svr_predict_all
    else:
        model_ml, all_ml ='xgboost', xgboost_train
        feature_drop=xgboost_drop
        demand_ml=xgboost_predict_all
 
    return demand_ml, all_ml, model_ml, feature_drop, xgboost_test, xgboost_train, xgboost_all, svr_test, svr_train, svr_all

# 預測                       
def forecast(raw_data, item_id):
    data_id=raw_data[raw_data['id'] == item_id]
    
    demand=data_id['sales'].reset_index(drop=True)
    demand.index += 1 # index start from 1
    
    plt.plot(data_id['week'], data_id['sales'])
    plt.title('Sales - %s (sales=%d)' % (item_id, data_id['sales'].sum()))
    #plt.savefig('%s_Sales.png' % item_id, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 比較 winter and holt   
    demand_winter, smape_winter=model_winter(data_id, demand)          
    demand_holt, smape_holt=model_holt(data_id, demand)
    
    # 以前80%(Train)歷史銷售量預測後20%(Test)銷售量，產出sMAPE
    smape_holt_test=smape_new(demand_holt['sales'][84:], demand_holt['forecast'][84:])
    smape_winter_test=smape_new(demand_winter['sales'][84:], demand_winter['forecast'][84:])
    smape_holt_train=smape_new(demand_holt['sales'][:84], demand_holt['forecast'][:84])
    smape_winter_train=smape_new(demand_winter['sales'][:84], demand_winter['forecast'][:84])
    
    #比較 "Holt's" & "Winter's"訓練集精準度，取sMAPE較低的模型
    if np.isnan(smape_winter) == True: # 如果winter結果為nan(無法計算)
        demand=demand_holt.copy()        
        model_exp="Holt's"
        smape_exp_train=smape_holt_train
    else:
        if smape_winter_train >= smape_holt_train:
            demand=demand_holt.copy()
            model_exp="Holt's"            
            smape_exp_train=smape_holt_train
        else:
            demand=demand_winter.copy()
            model_exp="Winter's"            
            smape_exp_train=smape_winter_train
       
    # 比較 xgboost & svr    
    demand_ml, all_ml, model_ml, feature_drop, xgboost_test, xgboost_train, xgboost_all, svr_test, svr_train, svr_all=machine_learning(data_id, demand)
    
    # 比較ml & 傳統方法 (testing data)
    if all_ml <= smape_exp_train:
        best_model='ML'
        final_smape=all_ml
        demand_final=demand_ml
    else:
        best_model='Exp'
        final_smape=smape_exp_train
        demand_final=demand
      
    return demand_final, data_id['sales'].sum(), final_smape, best_model, model_exp, model_ml, feature_drop, smape_holt_test, smape_holt_train, smape_holt, smape_winter_test, smape_winter_train, smape_winter, xgboost_test, xgboost_train, xgboost_all, svr_test, svr_train, svr_all

'''
主程式
'''
if __name__ == '__main__':

    raw_data, id102=rawdata_id()
    
    results_final=[]
    for n in range(len(id102)): 
        item_id=id102[n].astype(int) 
        # 預測銷售量
        predict_all, total_sales, final_smape, best_model, model_exp, model_ml, feature_drop, smape_holt_test, smape_holt_train, smape_holt, smape_winter_test, smape_winter_train, smape_winter, xgboost_test, xgboost_train, xgboost_all, svr_test, svr_train, svr_all=forecast(raw_data, item_id)

        result=item_id, total_sales, final_smape, best_model, model_exp, model_ml, feature_drop, smape_holt_test, smape_holt_train, smape_holt, smape_winter_test, smape_winter_train, smape_winter, xgboost_test, svr_test, xgboost_train, svr_train, xgboost_all, svr_all
        results_final.append(result)
              
    results_final_df=pd.DataFrame(results_final, columns=['id', 'sales', 'best_smape_test', 'best_model', 'model_exp', 'model_ml', 'feature_drop', 'smape_holt_test', 'smape_holt_train', 'smape_holt',
                                                          'smape_winter_test', 'smape_winter_train', 'smape_winter', 'xgboost_test', 'svr_test', 'xgboost_train', 'svr_train', 'xgboost_all', 'svr_all']) 
       
    results_final_df.to_csv('results102_20p.csv') # 20p表示測試集為20%的情況下得到的預測結果

     