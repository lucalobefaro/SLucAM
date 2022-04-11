function res = plot_keyframe(folder, idx)

    % Initialization
    base_filename = folder + "keyframe";
    image_name_filename = "_SLucAM_image_name.dat";
    landmarks_filename = "_SLucAM_landmarks.dat";
    poses_filename = "_SLucAM_poses.dat";
    edges_filename = "_SLucAM_edges.dat";
    image_points_filename = "_SLucAM_img_points.dat";

    % Load poses, image and image's points
    poses = load_poses(base_filename+int2str(idx)+poses_filename);
    landmarks = load_landmarks(base_filename+int2str(idx)+landmarks_filename);
    img_filename = load_img_filename(base_filename+int2str(idx)+image_name_filename);
    edges = load_edges(base_filename+int2str(idx)+edges_filename);
    img_points = load_image_points(base_filename+int2str(idx)+image_points_filename);

    % Set some prior parameter for plot
    set(gcf, 'Position', get(0, 'Screensize'));
    t = tiledlayout(3,4);

    % Plot the image
    nexttile(1, [1 1]);
    plot_image(img_filename);
    hold on;
    plot_image_points(img_points);
    title("Current measurement");
    
    % Plot pose graphs
    nexttile(5, [1,1]);
    plot_pose_graph(poses, "top");
    xlabel("x");
    ylabel("y");
    title("Pose graph (top view)");

    nexttile(9, [1,1]);
    plot_pose_graph(poses, "side");
    xlabel("x");
    ylabel("z");
    title("Pose graph (side view)");

    % Plot poses and landmarks
    nexttile(2, [3 3]);
    plot_poses(poses);
    hold on;
    plot_landmarks(landmarks);
    hold on;
    plot_edges(poses, edges, landmarks);
    hold on;
    set(gca,'XColor', 'none','YColor','none','ZColor','none');
    view([0 -80]);
    axis([-1.5 1.5 -1 1 -0.5 10])
    camproj('perspective');
    title("World");
    
    % Set some posterior parameters for plot
    t.TileSpacing = 'compact';
    t.Padding = 'compact';

end