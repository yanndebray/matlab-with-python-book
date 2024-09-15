# -*- coding: utf-8 -*-
'''
Created on Thu Jan  4 20:41:45 2018
Modified Jan 29 2019
Simplified June 16 2022
Updated Sept 15 2024
'''

# weather.py
import datetime
import json  
import urllib.request

BASE_URL = '/data/2.5/{}?q={},{}&appid={}'

def get_current_weather(city, country, apikey,**kwargs):  
    '''get current conditions in specified location
    appid='b1b15e88fa797225412429c1c50c122a1'
    get_current_weather('London','uk',appid,api='samples')'''

    # select between samples or prod api
    if 'api' in kwargs:
        if kwargs['api'] == 'samples':
            DOMAIN = "http://samples.openweathermap.org"
        else:
            print('not the right api arg')
            DOMAIN = "http://api.openweathermap.org"
    else:
        DOMAIN = "http://api.openweathermap.org"

    # Read current conditions
    try:
        # url = 'http://samples.openweathermap.org/data/2.5/weather?q=London,uk&appid=b1b15e88fa797225412429c1c50c122a1'
        url = DOMAIN + BASE_URL.format('weather',city,country,apikey)
        json_data = json.loads(urllib.request.urlopen(url).read())  
        return json_data 
    except urllib.error.URLError:
        # If weather API doesnt work
        print("API not available")


def parse_current_json(json_data):
    '''parse and extract json data from the current weather data''' 

    try:
        # select data of interest from dictionary
        weather_info = json_data['main']
        weather_info.update(json_data['wind'])
        weather_info.update(json_data['coord'])
        weather_info['city'] = json_data['name']
        # add current date and time
        weather_info['current_time'] = str(datetime.datetime.now())

    except KeyError as e:
        # use current dictionary (because it probably came from backup file) 
        try:
            # If this fails then the json_data didn't come from backup file
            json_data.pop('City')
            weather_info = json_data
        except:
            # print('Something else went wrong while parsing current json')
            raise e
    
    return weather_info


def get_forecast(city, country, apikey, **kwargs):
    '''get forecast conditions in specified location
    appid='b1b15e88fa797225412429c1c50c122a1'
    get_forecast('Muenchen','DE',appid,api='samples')'''

    # select between samples or prod api
    if 'api' in kwargs:
        if kwargs['api'] == 'samples':
            DOMAIN = "http://samples.openweathermap.org"
        else:
            print('not the right api arg')
            DOMAIN = "http://api.openweathermap.org"
    else:
        DOMAIN = "http://api.openweathermap.org"
    # get forecast    
    try:
        url = DOMAIN + BASE_URL.format('forecast',city,country,apikey)
        json_data = json.loads(urllib.request.urlopen(url).read())   
        return json_data 
    except: 
        print("API not available")


def parse_forecast_json(json_data):
    '''parse and extract json data from the weather forecast data'''     
    
    try:
        # parse forecast json data
        data = json_data['list']
        wind_keys = ['deg','speed']
        weather_info = {'current_time':[], 'temp':[], 'deg':[],
                        'speed':[], 'humidity':[], 'pressure':[]}
        for data_point in data[0:40]:
            for k in list(weather_info.keys())[1:]: #Taking a slice so we don't add the time
                weather_info[k].append(float(data_point['wind' if k in wind_keys else 'main'][k]))
            weather_info['current_time'].append(data_point['dt_txt'])
        return weather_info
        
    except:
        print('Something went wrong while parsing forecast json')


def parse_forecast_json2(json_data):
    '''parse and extract json data from the weather forecast data as array'''     

    import array
    # parse forecast json data
    try:
        data = json_data['list']
        # create arrays
        temp = array.array('f')
        pressure = array.array('f')
        humidity = array.array('f')
        speed = array.array('f')
        deg = array.array('f')
        date = []
        
        # loop over all and add to arrays
        for i in range(40):
            x1 = data[i]
            temp.append(x1['main']['temp'])
            pressure.append(x1['main']['pressure'])
            humidity.append(x1['main']['humidity'])
            speed.append(x1['wind']['speed'])
            deg.append(x1['wind']['deg'])
            date.append(x1['dt_txt'])
                       
        # create dictionary
        weather_info = dict(current_time=date,temp=temp,deg=deg,
                                speed=speed,humidity=humidity,pressure=pressure)
        return weather_info        
    except:
        print('Something went wrong while parsing forecast json as array')