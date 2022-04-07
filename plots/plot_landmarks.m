# Initialization
filename = "../data/datasets/tum_dataset_2/predicted_landmarks.dat";

# Open the file
file = fopen(filename, "r");

# Ignore the header
fgets(file); fgets(file);

# Read all the points
points = []; i=1;
while( (line = fgets(file)) != -1) 
    points(:,i) = [str2double(strsplit(line){1}); 
                str2double(strsplit(line){2}); 
                str2double(strsplit(line){3})];
    i++;
end
points

# Close the file
fclose(file);

# Visualize the points
plot3(points(1,:), points(2,:), points(3,:), 'b*', "linewidth", 2);
title("giovanni");
grid;
waitfor(gcf);