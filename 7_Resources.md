# 7.	Resources

## 7.1.	Getting Python packages on MATLAB Online
MATLAB Online is a great pre-configured environment to demo MATLAB with Python. And I’m not just saying this now that I’ve joined the MATLAB Online product team (mid 2023). I have been using it since 2022 to run workshops with Heather, either for large public events like MATLAB EXPO, or for customer dedicated hands-on workshops. This avoids wasting the first half hour of any conversation in setting up the MATLAB & Python environments (see chapter 3), as Python 3 is pre-installed there.
But there’s a catch: what always prevented me from using this nice online environment productively for bilingual workflows was the lack of Python packages, and a way to customize the environment. I obviously tried to find ways around this limitation, by uploading the packages sources as zipped files, and unzip it in MATLAB Online. However, for foundational packages like Numpy, it was already amounting for over 7,000 files to write to disk (only 50 Mb uncompressed). So, it takes a while…
Then recently, after looking at how ChatGPT Code Interpreter works (now called Advanced Data Analysis), I realized it would be possible to [retrieve Python wheels and install them](https://stackoverflow.com/questions/36132350/install-python-wheel-file-without-using-pip) on the remote machine. [Python wheels](https://realpython.com/python-wheels/)  are a way to distribute Python packages. They are similar to zip files, but they contain all of the information needed to install a package, including the code, the dependencies, and the metadata. Wheels are typically used with the pip package manager. 
Having understood that the pip wheel can be used as an executable, all you have to do to enable pip in MATLAB Online is to retrieve the pip wheel, and use the following system command: 
```matlab
>> !python pip.whl/pip --version
pip 23.3 from /MATLAB Drive/pip.whl/pip (python 3.10)
```
You can create a simple setup file, setuppip.m:
```matlab
piploc = "/tmp/";
pipwhlname = getwheel("pip");
copyfile(pipwhlname, piploc+"pip.whl")
disp("pip set up in folder: "+piploc)
```
And wrap the code that gets the wheel in a function:
```matlab
function whlname = getwheel(pkg)
url = "https://pypi.org/pypi/" + pkg + "/json";
jason = webread(url);
urlwhl = jason.urls(1).url;
[~, filename, fileext] = fileparts(urlwhl);
whlname = [filename fileext];
websave(whlname,urlwhl);
end
```