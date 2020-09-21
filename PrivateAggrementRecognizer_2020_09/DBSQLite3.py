#! /usr/bin/python3

import sqlite3

def init_db():
    conn = sqlite3.connect('./database/luckycat.db')
    conn.commit()
    conn.close()

def insert_data(data,str):
    conn = sqlite3.connect('./database/luckycat.db')
    c = conn.cursor()
    c.execute("INSERT INTO "+str+" VALUES(?)",(data,))
    conn.commit()
    conn.close()

def delete_data(data,str):
    conn = sqlite3.connect('./database/luckycat.db')
    c = conn.cursor()
    c.execute("DELETE FROM "+str+" WHERE DATA=(?)",(data,))
    conn.commit()
    print("TABLE '"+str+"' DELETE DATA SUCCESSFUL\n") 
    conn.close()

def output_result(str):
    conn = sqlite3.connect('./database/luckycat.db')
    print("TABLE '"+str+"':")
    c = conn.cursor()
    cursor = c.execute("SELECT DATA FROM "+str)
    for row in cursor:
        print(row[0])
    conn.close()

def insert_list(list,str):
    conn = sqlite3.connect('./database/luckycat.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS "+str+"(DATA INT NOT NULL)")
    for x in list:
        insert_data(x,str)
    print("TABLE '"+str+"' INSERT DATA SUCCESSFUL\n") 
