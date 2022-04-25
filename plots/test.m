% First camera
R1 = eye(3);
t1 = [0 0 0];
rigid_pose1 = rigid3d(R1,t1);
plotCamera('AbsolutePose', rigid_pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
hold on;

% Second camera
T2 = [0.99688351, 0.0039046779, 0.078791946, -0.081914835;
 -0.00398257, 0.99999177, 0.00083251297, -0.0080929631;
 -0.078788072, -0.0011436492, 0.99689066, -0.056784358;
 0, 0, 0, 1];
T2 = inv(T2);
R2 = T2(1:3,1:3);
t2 = T2(1:3,4)';
rigid_pose2 = rigid3d(single(R2),t2);
plotCamera('AbsolutePose', rigid_pose2, 'Opacity', 0, ...
    "Color", "red", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
waitforbuttonpress;