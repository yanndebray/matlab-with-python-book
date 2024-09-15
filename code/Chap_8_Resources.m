%[text] # 8\. Resources
%[text] ## 8\.1\.       Getting Python packages on MATLAB Online
%[text] MATLAB Online is a great pre\-configured environment to demo MATLAB with Python. And I’m not just saying this now that I’ve joined the MATLAB Online product team (mid 2023). I have been using it since 2022 to run workshops with Heather, either for large public events like MATLAB EXPO, or for customer dedicated hands\-on workshops. This avoids wasting the first half hour of any conversation in setting up the MATLAB & Python environments (see [<u>chapter 3</u>](about:blank<#_Set-up_MATLAB_and_1%3E)), as Python 3 is pre\-installed there.
%[text] But there’s a catch: what always prevented me from using this nice online environment productively for bilingual workflows was the lack of Python packages, and a way to customize the environment. I obviously tried to find ways around this limitation, by uploading the packages sources as zipped files, and unzip it in MATLAB Online. However, for foundational packages like Numpy, it was already amounting for over 7,000 files to write to disk (only 50 Mb uncompressed). So, it takes a while…
%[text] Then recently, after discussing with my colleagues in the MATLAB Online team, they suggested a very simple approach to retrieve pip from a single script:
websave("get-pip.py","https://bootstrap.pypa.io/get-pip.py");
!python get-pip.py
!python -m pip --version
%%
%[text] You can now simply install a package like numpy as such:
!python -m pip install numpy

%[appendix]
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
