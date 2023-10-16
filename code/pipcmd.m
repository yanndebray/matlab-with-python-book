function pipcmd(cmd)
% cmd = "--version"
% pipcmd(cmd)
piploc = "/tmp/";
% piploc = "/MATLAB Drive/Python";
if ~exist(piploc+"pip.whl","file")
    setuppip
end
system("python "+piploc+"/pip.whl/pip " + cmd);
end