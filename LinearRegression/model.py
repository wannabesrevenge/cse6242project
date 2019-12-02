# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:06:07 2019

@author: Asher
"""

import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

games =  pd.read_csv('in/game_stats.csv')
defense = pd.read_csv('out/defense.csv')
offense = pd.read_csv('out/offense.csv')
kicking = pd.read_csv('out/kickers.csv')

teams = np.unique(games['home_team'].values)

team_stats = {}

for team in teams:
    
    def_stat = 0
    off_stat = 0
    kick_stat=0

    for index, row in  defense.iterrows():
        
        if row['team'] == team:
            
            def_stat = def_stat + row['stat_total']
            
    for index ,row in offense.iterrows():
        
        if row['team'] == team:
            
            off_stat = off_stat + row['stat_total']
            
    for index ,row in kicking.iterrows():
        
        if row['team'] == team:
            
            kick_stat = kick_stat + row['stat_total']
            
    team_stats[team] =[off_stat,def_stat,kick_stat]
    

test_games = games[games['nfl_id'] > 2018123014] 
train_games = games[games['nfl_id'] <= 2018123014]

train_results = pd.DataFrame(columns=['home', 'home_score', 'away','away_score', 'winner', 'final_score'])


for index, row in train_games.iterrows():
    
    final_score = row['home_score'] - row['away_score']
    
    winner = 'tie'
    
    if final_score > 0 :
        
        winner = row['home_team']
    elif  final_score < 0:
        winner = row['away_team']
        
    train_results = train_results.append({'home': row['home_team'], 'home_score': row['home_score'], 'away': row['away_team'], 'away_score' : row['away_score'], 'winner' : winner, 'final_score': final_score}, ignore_index=True)

test_results = pd.DataFrame(columns=['home', 'home_score', 'away','away_score', 'winner', 'final_score'])


for index, row in test_games.iterrows():
    
    final_score = row['home_score'] - row['away_score']
    
    winner = 'tie'
    
    if final_score > 0 :
        
        winner = row['home_team']
    elif  final_score < 0:
        winner = row['away_team']
        
    test_results = test_results.append({'home': row['home_team'], 'home_score': row['home_score'], 'away': row['away_team'], 'away_score' : row['away_score'], 'winner' : winner, 'final_score': final_score}, ignore_index=True)        
          
            
x_train = []
y_train = []
scaler = StandardScaler()

for index, row in  train_results.iterrows():
    
    home = team_stats[row['home']]
    away = team_stats[row['away']]
    
    x_train.append([home[0]-away[1],  away[0]-home[1],home[2],away[2]])
    y_train.append(row['final_score'])
    
x_train= scaler.fit_transform(x_train)
    
x_test = []
y_test = []

for index, row in  test_results.iterrows():
    
    home = team_stats[row['home']]
    away = team_stats[row['away']]
    
    x_test.append([home[0]-away[1],  away[0]-home[1],home[2],away[2]])
    y_test.append(row['final_score'])

x_test= scaler.fit_transform(x_test)

reg = LinearRegression().fit(x_train, y_train)
score =reg.score(x_train, y_train)

y_pred=  reg.predict(x_test)

acc = 0.0
num_games =len(y_pred)

for i in range(num_games):
    
    if y_pred[i]*y_test[i] > 0:
        acc=acc+1
    elif y_pred[i] == y_test[i] and y_test[i] == 0:
        acc=acc+1
        
        
acc= acc/num_games  

params= reg.coef_

print(params)
print(acc)
 
    


