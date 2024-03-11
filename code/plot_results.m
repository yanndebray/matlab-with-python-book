function figHndl = plot_results(res, plotTitle)
%PLOT_RESULTS Plot results from call_sim_the_model

figHndl = figure; hold on; cols = colororder;


plot(res{1}.vertical_disp.Time, res{1}.vertical_disp.Data, 'Color', cols(1,:), ...
    'DisplayName', 'vertical displacement: 1st sim with default Mb value');
plot(res{2}.vertical_disp.Time, res{2}.vertical_disp.Data, 'Color', cols(2,:), ...
    'DisplayName', 'vertical displacement: 2nd sim with new Mb value');


hold off; grid; 

title(plotTitle,'Interpreter','none');
set(get(gca,'Children'),'LineWidth',2);
legend('Location','southeast');

end

% To call this MATLAB function from Python make sure to install the correct version of Python
% https://www.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html?s_tid=srchtitle_Configure%20Your%20System%20to%20Use%20Python_1
% https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf
