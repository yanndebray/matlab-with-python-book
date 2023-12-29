function TT = preprocess(TT)
	% Retime timetable
	TT = retime(TT,"regular","linear","TimeStep",hours(1));
	% Smooth input data
	TT = smoothdata(TT,"movmean","SmoothingFactor",0.25);
end