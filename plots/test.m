% First camera
R1 = eye(3);
t1 = [0 0 0];
rigid_pose1 = rigid3d(R1,t1);
plotCamera('AbsolutePose', rigid_pose1, 'Opacity', 0, ...
    "Color", "blue", "Size", 0.05);
hold on;

% Second camera
T2 = [0.99990702, -0.01340469, 0.0025055115, 0.037949942;
 0.013341397, 0.99962854, 0.023767024, -0.99824089;
 -0.0028231721, -0.023731394, 0.99971443, 0.045551632;
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