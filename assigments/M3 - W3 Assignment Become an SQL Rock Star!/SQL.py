import sqlite3
import pandas as pd
import os
import re


os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

# Find all the table names:
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = cursor.fetchall()

# SQL to DataFrame - Run Query
df_albums = pd.read_sql_query('SELECT * FROM albums', conn)
df_sqlite_sequence = pd.read_sql_query('SELECT * FROM sqlite_sequence', conn)
df_artist = pd.read_sql_query('SELECT * FROM artists', conn)
df_customers = pd.read_sql_query('SELECT * FROM customers', conn)
df_employees = pd.read_sql_query('SELECT * FROM employees', conn)
df_genres = pd.read_sql_query('SELECT * FROM genres', conn)
df_invoices = pd.read_sql_query('SELECT * FROM invoices', conn)
df_invoice_items = pd.read_sql_query('SELECT * FROM invoice_items', conn)
df_media_types = pd.read_sql_query('SELECT * FROM media_types', conn)
df_playlists = pd.read_sql_query('SELECT * FROM playlists', conn)
df_playlist_track = pd.read_sql_query('SELECT * FROM playlist_track', conn)
df_tracks = pd.read_sql_query('SELECT * FROM tracks', conn)
df_sqlite_stat1= pd.read_sql_query('SELECT * FROM sqlite_stat1', conn)

# MERGE the two dataframes together
df_dataset = pd.merge(df_albums, df_artist, how="inner", on='ArtistId')

# Loop through all the database tables
df = []
for table_name in table_names:
    # SQL to data frame for table name
    match = re.search(r"^[a-zA-Z_\d]+$", table_name[0])
    name = match[0]
    if name != "sqlite":
        df = pd.read_sql_query('SELECT * FROM {}'.format(name), conn)
    
    # print the columm names for the table:
    print(name)
    print(df.columns)
    print()
