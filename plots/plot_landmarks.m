function res = plot_landmarks(landmarks)

    for i = 1:size(landmarks, 2)

        scatter3(landmarks(1,i), ...
                landmarks(2,i), ...
                landmarks(3,i));
        hold on;

    end

end