function h = my_subplotter(i)
    [c,r] = ind2sub([3 1], i);
    if(i == 1)
        ax = subplot('Position', [(c-1)/3, 1-(r), 1/3, 1]);
    else
        ax = subplot('Position', [(c-1)/3, 1-(r), (1/3)*2, 1]);
    end
end