function whlname = getwheel(pkg)
url = "https://pypi.org/pypi/" + pkg + "/json";
% Fetch the JSON data
jason = webread(url);
% Parse the JSON and extract the desired value
urlwhl = jason.urls(1).url;
% !wget https://files.pythonhosted.org/packages/50/c2/e06851e8cc28dcad7c155f4753da8833ac06a5c704c109313b8d5a62968a/pip-23.2.1-py3-none-any.whl
% urlwhl = "https://files.pythonhosted.org/packages/50/c2/e06851e8cc28dcad7c155f4753da8833ac06a5c704c109313b8d5a62968a/pip-23.2.1-py3-none-any.whl"
[~, filename, fileext] = fileparts(urlwhl);
whlname = [filename fileext];
% whlname = "/tmp/"+whlname;
websave(whlname,urlwhl);
end