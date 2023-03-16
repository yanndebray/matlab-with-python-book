###########################################################################################	

"""
dataLib.py imports 1-minute bars from Coinbase Pro.  Note, the API does
not fetch the first minute specified by the start date so the times 
span (start, stop].  Updated on 10/1/2019 with Python 3.7

Example of a RESTful call dataLib will make on the user's behalf:
https://api.pro.coinbase.com/products/ETH-USD/candles?start=2019-10-02T12:00:00Z&end=2019-10-02T17:00:00Z&granularity=60

To call from Python:
>>import dataLib
>>jsonData = dataLib.getPriceData('ETH', '2019-10-02T12:00:00Z', '2019-10-02T17:00:00Z')
>>data = dataLib.parseJson(jsonData, [0,1,4]) 
"""

###########################################################################################	

def getPriceData(coin, start, stop):
	# returns back a python list containing the historical data
	# of the cryptocurrency 'product' for the period of time 
	# ('start', 'stop'].
	
	import urllib.request
	import json
	import os

	# website we want to pull data from 
	hostname = 'api.pro.coinbase.com'
	
	# all cryptocurrency products returned in USD currency
	product = coin + '-USD'
	
	# granularity is in seconds, so we are getting 1-minute bars
	granularity = '60'

	# returns back: [date, low, high, open, close, volume]
	url = 'https://' + hostname + '/products/' + product + '/candles?start=' + start + '&end=' + stop + '&granularity=' + granularity

	if isOnline(hostname):
		# execute call to the website
		urlRequest = urllib.request.Request(url, data=None, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
		response = urllib.request.urlopen(urlRequest)
		html = response.read()

		# python 3.x requires decoding from bytes to string 
		data = json.loads(html.decode())		
	else:
		# we are not online, so for demo purposes lets load in 
		# a saved dataset 
		print("Not online.  Using savedData.json instead.")		
		thisDirectory = os.path.dirname(os.path.realpath(__file__))
		file = open(thisDirectory + "\savedData.json", 'r')
		data = json.loads(file.read())
		file.close()
		
		# return back data for the desired cryptocurrency only
		data = data[coin];
	return data

###########################################################################################	

def parseJson(jsonData, selectedColumns):
	# returns back a tuple where the first output is a numpy array
	# consisting of only the selectedColumns in the dataset. the
	# second output is a list of the column header names.
	
	import numpy as np

	# define the column header names (as specified in the Coinbase API)
	columnNames = np.array(['Date', 'Low', 'High', 'Open', 'Close', 'Volume'])
	
	# convert list to an Nx6 array
	data = np.array(jsonData)
		
	# ensure selectedColumns are integers for indexing into columnNames 
	filter = np.array(selectedColumns, dtype=int)
	
	# slice data to user's selection
	selectedColumnNames = columnNames[filter]
	selectedData = data[:, filter]
	
	# returns back the time and close price
	return (selectedData, selectedColumnNames.tolist())
	
###########################################################################################	
	
def isOnline(hostname):
	# returns boolean on if we are connected to the internet 
	# (e.g. can reach the hostname specified).  used internally
	# inside getPriceData

	import socket

	try:
		# check if hostname can be resolved via DNS lookup
		host = socket.gethostbyname(hostname)
    
		# check if host is reachable 
		port = 80
		timeout = 2
		theConnection = socket.create_connection((host, port), timeout)
		theConnection.close()
		
		# no errors, so we are connected to the internet 
		return True
	except:
		# error was raised, not connected to the internet
		pass
	return False