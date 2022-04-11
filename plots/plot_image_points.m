function res = plot_image_points(points)

    for i = 1:size(points, 2)

        scatter(points(1,i), ...
                points(2,i));
        hold on;

    end

end