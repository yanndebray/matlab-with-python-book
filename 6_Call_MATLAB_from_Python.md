# 6.	Call MATLAB from Python

If you are a Python user and you are wondering why you should consider MATLAB, this chapter is probably a better entry point into this book. One of my favorite colleagues, Lucas Garcia – Deep Learning Product Manager – tried to answer this question at a Python Meetup  in Madrid:
- Facilitate AI development by using a simplified MATLAB workflow
- Need functionality available in MATLAB (e.g. Simulink)
- Leverage the work from the MATLAB community

But first let’s start with a few basics on how to call MATLAB from Python.

## 6.1.	Getting started with the MATLAB Engine API for Python
First, make sure you have the MATLAB Engine for Python installed (as described in section 3.8).
In your python script or jupyter notebook, the first statement you will need to enter in order to load the MATLAB Engine package is:
```python
>>> import matlab.engine
```
You have then two options to interact with MATLAB from Python:
1.	Start a new session (in batch or interactive)
By default, the engine will be started in batch with the “-nodesktop” mode:
```python
>>> m = matlab.engine.start_matlab()
```
![](media/image100.png)
(this is the contextual help provided by VSCode when you enter the function)
If you wish to have the MATLAB Desktop apparent to visualize which values are stored in the workspace or to debug interactively with the console, you can specify the “desktop” argument:
```python
>>> m = matlab.engine.start_matlab("-desktop")
```
2.	Connect to an existing session
First you need to start MATLAB manually. For convenience, it’s easier to have the MATLAB Desktop and your Python development environment (Jupyter or VSCode) open side by side. To share the MATLAB Engine session, simply type inside of MATLAB:
```matlab
>> matlab.engine.shareEngine
```
You can also request the name of the MATLAB Engine session, in case Python doesn’t find it automatically:
```matlab
>> matlab.engine.engineName
ans =
    'MATLAB_11388'
```    
Then on the Python side, enter the following command:
```python
>>> m = matlab.engine.connect_matlab()
```
![](media/image101.png)

(this is the contextual help provided by VSCode when you enter the function)
If Python does not find automatically the running session, you can enter the engine name requested previously in MATLAB (`'MATLAB_11388'`).

## 6.2.	Facilitate AI development by using MATLAB Apps

### 6.2.1. Data Cleaner App

The first step in an AI pipeline is often to clean the data. This process requires some level of interactivity for the data analyst to understand which variables she is manipulating. Once the input format of the data is fixed, this process can be automated to scale it to the whole dataset.
Let’s take an example with the weather data from chapter 2. In this first example, we will start MATLAB in `-nodesktop` mode (which is the default mode for the engine). In the next two sections, we will use the `-desktop` mode to show how to use the MATLAB desktop to interact with the data, but also connect to an already running MATLAB session.
Set up the environment
```
!git clone https://github.com/hgorr/weather-matlab-python
```
```python
import matlab.engine
m = matlab.engine.start_matlab()
m.cd('weather-matlab-python') # returns the previous dir location 
# m.cd('..')
```
```text:Output
'C:\\Users\\ydebray\\Downloads\\python-book-github'
```
```python
m.pwd()
```
```text:Output
'C:\\Users\\ydebray\\Downloads\\python-book-github\\weather-matlab-python'
```
```python
# Make sure that your Python interpreter follows along
import os
os.getcwd()
os.chdir('weather-matlab-python')
# os.chdir('..')
```
**Retrieve Weather Data**
```python
import weather
appid ='b1b15e88fa797225412429c1c50c122a1'
json_data = weather.get_forecast('Muenchen','DE',appid,api='samples')
data = weather.parse_forecast_json(json_data)
data.keys()
```
```text:Output
dict_keys(['current_time', 'temp', 'deg', 'speed', 'humidity', 'pressure'])
print(len(data['temp']))
data['temp'] [0:5]
```
```text:Output
36
[286.67, 285.66, 277.05, 272.78, 273.341]
```
```python
t = matlab.double(data['temp'])
t
```
```text:Output
matlab.double([[286.67,285.66,277.05,272.78,273.341,275.568,276.478,276.67,278.253,276.455,275.639,275.459,275.035,274.965,274.562,275.648,277.927,278.367,273.797,271.239,269.553,268.198,267.295,272.956,277.422,277.984,272.459,269.473,268.793,268.106,267.655,273.75,279.302,279.343,274.443,272.424]])
```
**Format into a Timetable**
```python
# Transform into a timetable for data cleaning
m.workspace['data'] = data
m.eval("TT = timetable(datetime(string(data.current_time))',cell2mat(data.temp)','VariableNames',{'Temp'})",nargout=0)
m.who()
```
```text:Output
['TT', 'data']
```
**Interact manually with the app**
```python
m.dataCleaner(nargout=0)
```
The app will appear, with a blank canvas, giving you the option to import data. Select the timetable from your workspace.

![](media/image117.png)

![](media/image118.png)
 		 
This will open your data into the app main view:

![](media/image119.png)
 
In the left panel, you can select the variables that you want to visualize and manipulate. 


We will select the Time variable, and use the **Retime Timetable** cleaning method:

![](media/image120.png)

This will display options in the right panel, where we will specify the new sampling, Time step: 1 (hour). Once you are happy with the results, click accept (bottom right).

![](media/image121.png)
 
You can see the results of this transformation on your data by changing the tabs in the center panel:

![](media/image122.png)
 
The second transformation we will operate on this is **Smooth Data**. 

![](media/image123.png)

You can select the smoothing method (we will stick with the default moving average) and play around with the smoothing factor.

![](media/image124.png)

**Export the cleaning steps**
Once you are happy with the way your data looks, you can save your manual operations as a function that you will apply to any new weather data that comes in, to automate the preprocessing.

![](media/image125.png) 

[preprocess.m](code/preprocess.m)
```matlab
function TT = preprocess(TT)
	% Retime timetable
	TT = retime(TT,"regular","linear","TimeStep",hours(1));
	% Smooth input data
	TT = smoothdata(TT,"movmean","SmoothingFactor",0.25);
end
```
You can call this function from Python, and test that it works.
```python
TT = m.workspace['TT']
TT2 = m.preprocess(TT)
m.parquetwrite("data.parquet",TT2,nargout=0)
import pandas as pd
pd.read_parquet('data.parquet').plot(y='Temp')
```

![](media/image126.png)


### 6.2.2. Regression and Classification Learner Apps
We will take the Boston housing example that is part of the Scikit-Learn sample datasets to call MATLAB from Python. 
Open a Jupyter notebook and connect to a running MATLAB session from Python:
```python
import matlab.engine
m = matlab.engine.connect_matlab()
```
Retrieve the dataset
```python
import sklearn.datasets
dataset = sklearn.datasets.load_boston()
dataset.keys()
```
```text
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])
```
```python
data = dataset['data']
target = dataset['target']
```
Depending on the version of MATLAB you are you using, you might require converting the data and target arrays to MATLAB double:
-	Before 22a, Numpy arrays were not accepted, so you need to translate them to lists.
-	In 22a, Numpy arrays can be passed into MATLAB object constructor (double, int32, …).
-	From 22b, Numpy arrays can be passed directly into MATLAB functions.

```python
# Before 22a
X_m = matlab.double(data.tolist())
Y_m = matlab.double(target.tolist())
# In 22a
X_m = matlab.double(data)
Y_m = matlab.double(target)
# From 22b
X_m = data
Y_m = target
# Call the regression learner app with data coming from Python
m.regressionLearner(X_m,Y_m,nargout=0)
```
The session is automatically created in the Regression Learner, with the passed data:

![](media/image105.png)

You have several models and categories to choose from:

![](media/image106.png)

You can visualize certain indicators during the training:

![](media/image107.png)
![](media/image108.png)
![](media/image109.png)


Once you are happy with one of the models you’ve trained, you can generate a function or export it:

![](media/image110.png)

Informations are shared in the MATLAB Command Window: 

Variables have been created in the base workspace. Structure 'trainedModel' exported from Regression Learner. 
To make predictions on a new predictor column matrix, X: 
    yfit = trainedModel.predictFcn(X) 
For more information, see [How to predict using an exported model](https://www.mathworks.com/help/stats/export-regression-model-to-predict-new-data.html).


Finally, you can retrieve the model to assign the prediction function to a variable in Python:
```python
model = m.workspace['trainedModel']
m.fieldnames(model)
```
```text
['predictFcn', 'RegressionTree', 'About', 'HowToPredict']
```
```python
predFcn = model.get('predictFcn')
```
This way, you can test the model directly from within Python:
```python
X_test = data[0]
y_test = target[0]
X_test,y_test
```
```text
(array([6.320e-03, 1.800e+01, 2.310e+00, 0.000e+00, 5.380e-01, 6.575e+00,
        6.520e+01, 4.090e+00, 1.000e+00, 2.960e+02, 1.530e+01, 3.969e+02,
        4.980e+00]),
 24.0)
```
```python
m.feval(predFcn,X_test)
```
```text
23.46666666666667
```

You can iterate and test another model to see if the predictions are closer to the test target.

### 6.2.3.	Image Labeler App
Data preparation is key in developing Machine Learning and Deep Learning applications. No matter how much effort you put into your ML model, it will likely perform poorly if you didn’t spend the right time in preparing your data to be consumed by your model.
In this example, we start with a set of images to label for a Deep Learning application.
```python
import os
cwd = os.getcwd()
vehicleImagesPath = os.path.join(cwd, "..", "images", "vehicles", "subset")
```
We then start the MATLAB Engine API for Python and open the Image Labeler App, passing the location of the images as an input:
```python
import matlab.engine
m = matlab.engine.connect()
m.imageLabeler(vehicleImagesPath, nargout=0)
```

![](media/image112.png)

*Note that because the App returns no output arguments back to Python, you need to specify `nargout=0` (number of output arguments equals 0).*

Now you can interactively create a new ROI Label:

![](media/image113.png)

and start to manually label your vehicles:

![](media/image114.png)

This process is rather tedious, especially considering that the number of required labeled images for the problem might be significant for Deep Learning workflows. Thus, the following shows how to facilitate labeling by automating (or semi-automating) the labeling process. After selecting the images you would like to automatically label, you can choose among various algorithms (*ACF vehicle detector, ACF people detector*, or import your custom detector). In this particular case, after choosing *ACF vehicle detector*, the selected images are automatically labeled. Earlier I mentioned that the process is semi-automated, as it might not detect all vehicles, or you might want to correct some bounding boxes before exporting your results.

![](media/image115.png)

Finally, export your labeling process as a MATLAB table to continue your work back in Python:

![](media/image116.png)
 
Back in Python, gather the variables you are interested in:
```python
imageFilename = m.eval("gTruth.imageFilename")
labels = m.eval("gTruth.vehicle")
```
and put them into a convenient form to continue your work:
```python
import pandas as pd
import numpy as np
# Bring data to convenient form as DataFrame
labels = [np.array(x) for x in labels]
df = pd.DataFrame({"imageFileName":imageFilename, "vehicle":labels})
```
Labeled data is now conveniently shaped into a DataFrame with information regarding file location and bounding boxes for each vehicle, and can be easily accessed:
```python
df.iloc[[13]]
m.exit()
```