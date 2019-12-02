# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:45:03 2019

@author: Asher
"""

import pandas as pd
import numpy as np 

data = pd.read_csv('in/offense_stats.csv')

player_games=[]
players= []

for index, row in data.iterrows():
    
    stats  = [row['pass_touchdowns'],row['recv_touchdowns'],row['rush_touchdowns']]
    game = [row['name'],row['team'], row['game_id'] ,stats]
    player = row['name']
        
    player_games.append(game)
    players.append(player)
    
players = np.unique(players,axis=0)

play = []
for i in range(len(players)):
    player_stats= []
    for game in player_games:
       
        if players[i] == game[0]:
            player_stats.append(game[1:])
            
    play.append([players[i],player_stats])

num_stats =3

offense = pd.DataFrame(columns=['posistion', 'name', 'team','stat_total', 'last_game'])

print(play[0])
for player in play:
    
    name = player[0]
    
    index = 0
    
    transfers = 0
    
    kicks_none = 0
    xp_none = 0
    
    last_game = 0
    

    
    transfer_dates = []
    transfer_index =[]

    old_team  = player[1][0][0]
    all_teams = [old_team]
    
    stats = [0.0,0.0,0.0]
    
    for game in player[1]:
        
        if game[0] != old_team:
            transfers = transfers + 1
            transfer_dates.append(game[1])
            transfer_index.append(index)
            all_teams.append(game[0])
            old_team = game[0]
            
        if game[1] > last_game:
            last_game = game[1]
            
        if stats[0] == 0:    
            stats[0] = game[2][0]
        else:  
            stats[0] = (index*stats[0]+1.0001*game[2][0])/(index+1.0001)
            
        if stats[1] == 0:    
            stats[1] = game[2][1]
        else:  
            stats[1] = (index*stats[1]+1.0001*game[2][1])/(index+1.0001)
            
        if stats[2] == 0:    
            stats[2] = game[2][2]
        else:  
            stats[2] = (index*stats[2]+1.0001*game[2][2])/(index+1.0001)
        
        index = index+1
    
            

    stat_total = np.sum(stats)/num_stats
    offense = offense.append({'posistion': 'o', 'name': name, 'team': old_team, 'stat_total' : stat_total, 'last_game' : last_game}, ignore_index=True)
    
offense.to_csv('out/offense.csv',index=False)      
        
    
        
        
    
        
        
