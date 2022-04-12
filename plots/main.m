% Initialization
warning('off');
folder = "../results/";
results_folder = "./images/";
n_keyframes = 196;


% create an invisible figure
f = figure('visible', 'off');

% Plot all the keyframes
for i = 0:n_keyframes
    
    % Plot the current keyframe
    plot_keyframe(folder, i);
    %
    % Save the plot
    print(f, '-djpeg', sprintf(results_folder+'image_%03d.jpg', i)); 
    
    % Clear the figure
    clf(f);

    disp("Keyframe " + int2str(i) + "/" + int2str(n_keyframes) + " saved.");

end

% Close the figure
delete(f);


%{
f = figure('visible', 'on');
plot_keyframe(folder, i);
%}
