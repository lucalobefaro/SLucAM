% First camera
R1 = eye(3);
t1 = [0 0 0];
rigid_pose1 = rigid3d(R1,t1);
plotCamera('AbsolutePose', rigid_pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
hold on;

% Second camera
T2 = [0.99994689, -0.0098595172, 0.0030167834, -7.243696e-05;
 0.0098022334, 0.99978173, 0.018446803, 0.058590002;
 -0.0031980011, -0.018416137, 0.99982536, -0.081038296;
 0, 0, 0, 1];
T2 = T2;
R2 = T2(1:3,1:3);
t2 = T2(1:3,4)';
rigid_pose2 = rigid3d(single(R2),t2);
plotCamera('AbsolutePose', rigid_pose2, 'Opacity', 0, ...
    "Color", "red", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
view([0 -80]);
camproj('perspective');
waitforbuttonpress;