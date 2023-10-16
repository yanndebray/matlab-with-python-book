piploc = "/tmp/";
pipwhlname = getwheel("pip");
copyfile(pipwhlname,piploc+"pip.whl")
% copyfile("/tmp/"+pipwhlname,"/tmp/pip.whl")
disp("pip set up in folder: "+piploc)