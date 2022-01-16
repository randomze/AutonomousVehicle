% Setup:
% Add `matlab.engine.shareEngine` to `startup.m` file (https://www.mathworks.com/help/matlab/ref/startup.html)
% - this makes it so whenever matlab opens, its engine becomes available to
% subsequent python calls using the engine API

% Run sim.py - this generates variables :
%       car         (struct) : contains car description
%       road        (struct) : contains road description
%       sim_ic      (struct) : contains simulation initial conditions
%       road_edges  (logical 2d array) : contains binary representation of road edges
%       road_img    (logical 2d array) : contains binary representation of road
%       blocks      (double Nx2 array) : contains position (x,y) of
%                                       bottomleft corner of all blocks
%                                       composing the road_edges

% Now you can run this script

addpath('matlab_utils\');

x = -12.77;
y = 60;
window = 20;

% generate and plot car
figure(1);
clf;
car_r = car_repr(x, y, pi/4, pi/6, car.L, car.Lr, car.Lf, car.d, car.r, 0.1); 
plot_car(car_r, {'-b', 'LineWidth', 1.2});

% select and plot blocks
ad_blocks = admissible_blocks(blocks, [x-window/2, x+window/2], [y-window/2, y+window/2]); 
plot_blocks(ad_blocks, road.meters_per_pixel); 
axis equal;

% actually only need to check for collisions between blocks in neighborhood
% of car and car (the following is already too much):
tic; collide = car_colliding_blocks(car_r, blocks, road.meters_per_pixel, [x, y], car.Lr+car.L+car.Lf+car.d)
toc % on the order of ms

% check collision between blocks in display and car
tic; collide = car_colliding_blocks(car_r, blocks, road.meters_per_pixel, [x, y], window)
toc % 2-10 times slower
% equivalent to
% collide = car_colliding_blocks(car, ad_blocks, meters_per_pixel)

% SLOW alternative is checking collision for all blocks
% tic; collide = car_colliding_blocks(car, blcks, meters_per_pixel);
% toc % >1000 times slower
clear ad_blocks collide blocks car_r window x y

