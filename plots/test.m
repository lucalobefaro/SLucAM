% First camera
R1 = eye(3);
t1 = [0 0 0];
rigid_pose1 = rigid3d(R1,t1);
plotCamera('AbsolutePose', rigid_pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
hold on;

% Second camera
T2 = [-0.99419576, 0.098383203, -0.043542925, -0.081914835;
 0.095356554, 0.61835223, -0.7800945, -0.0080929631;
 -0.049823351, -0.77971864, -0.62414461, -0.056784358;
 0, 0, 0, 1];
T2 = T2;
R2 = T2(1:3,1:3);
t2 = T2(1:3,4)';
rigid_pose2 = rigid3d(single(R2),t2);
plotCamera('AbsolutePose', rigid_pose2, 'Opacity', 0, ...
    "Color", "red", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
waitforbuttonpress;