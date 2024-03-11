function res = sim_the_model(args)
% Utility function to simulate a Simulink model with the specified parameters.
% 
% Inputs:
%    StopTime: simulation stop time, default is nan
%    TunableParameters:
%       A struct where the fields are the tunanle referenced
%       workspace variables with the values to use for the
%       simulation.
%    
%    Values of nan or empty for the above inputs indicate that sim should
%    run with the default values set in the model.
% 
% Outputs:
%    res: A structure with the time and data values of the logged signals.

arguments
    args.StopTime (1,1) double = nan
    args.TunableParameters = []
end

    %% Create the SimulationInput object
    si = Simulink.SimulationInput('suspension_3dof');
    %% Load the StopTime into the SimulationInput object
    if ~isnan(args.StopTime)
        si = si.setModelParameter('StopTime', num2str(args.StopTime));
    end
    
    %% Load the specified tunable parameters into the simulation input object
    if isstruct(args.TunableParameters) 
        tpNames = fieldnames(args.TunableParameters);
        for itp = 1:numel(tpNames)
            tpn = tpNames{itp};
            tpv = args.TunableParameters.(tpn);
            si = si.setVariable(tpn, tpv);
        end
    end

    %% call sim
    so = sim(si);
    
    %% Extract the simulation results
    % Package the time and data values of the logged signals into a structure
    res = extractResults(so,nan);

end % sim_the_model_using_matlab_runtime

function res = extractResults(so, prevSimTime)
    % Package the time and data values of the logged signals into a structure
    ts = simulink.compiler.internal.extractTimeseriesFromDataset(so.logsout);
    for its=1:numel(ts)
        if isfinite(prevSimTime)
            idx = find(ts{its}.Time > prevSimTime);
            res.(ts{its}.Name).Time = ts{its}.Time(idx);
            res.(ts{its}.Name).Data = ts{its}.Data(idx);
        else
            res.(ts{its}.Name).Time = ts{its}.Time;
            res.(ts{its}.Name).Data = ts{its}.Data;
        end
    end
end
