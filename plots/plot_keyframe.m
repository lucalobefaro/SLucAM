function res = plot_keyframe(folder, idx)

    % Initialization
    base_filename = folder + "keyframe";
    image_name_filename = "_SLucAM_image_name.dat";
    landmarks_filename = "_SLucAM_landmarks.dat";
    poses_filename = "_SLucAM_poses.dat";
    edges_filename = "_SLucAM_edges.dat";
    image_points_filename = "_SLucAM_img_points.dat";

    % Load poses and points
    poses = load_poses(base_filename+int2str(idx)+poses_filename);
    landmarks = load_landmarks(base_filename+int2str(idx)+landmarks_filename);
    img_filename = load_img_filename(base_filename+int2str(idx)+image_name_filename);
    edges = load_edges(base_filename+int2str(idx)+edges_filename);
    img_points = load_image_points(base_filename+int2str(idx)+image_points_filename);

    % Plot them
    set(gcf, 'Position', get(0, 'Screensize'));
    my_subplotter(2);
    plot_poses(poses);
    hold on;
    plot_landmarks(landmarks);
    hold on;
    plot_edges(poses, edges, landmarks);
    hold on;
    
    % Set some view parameter
    view([0 -80]);
    axis([-1.5 1.5 -1 1 -0.5 10])
    camproj('perspective');
    set(gca,'XColor', 'none','YColor','none','ZColor','none');

    % Plot the image
    my_subplotter(1);
    plot_image(img_filename);
    hold on;
    plot_image_points(img_points);

end