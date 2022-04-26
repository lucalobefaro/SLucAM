% First camera
R1 = eye(3);
t1 = [0 0 0];
rigid_pose1 = rigid3d(R1,t1);
plotCamera('AbsolutePose', rigid_pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
hold on;

% Second camera
T2 = [0.99822211, 0.00098778307, 0.059595123, 0.98959762;
 -0.0012292154, 0.99999112, 0.0040145586, -0.046543546;
 -0.059590615, -0.0040805642, 0.99821454, -0.13612621;
 0, 0, 0, 1];
T2 = inv(T2);
R2 = T2(1:3,1:3);
t2 = T2(1:3,4)';
rigid_pose2 = rigid3d(single(R2),t2);
plotCamera('AbsolutePose', rigid_pose2, 'Opacity', 0, ...
    "Color", "red", "Size", 0.1);
xlabel("x"); ylabel("y"); zlabel("z");
view([0 -80]);
camproj('perspective');
waitforbuttonpress;