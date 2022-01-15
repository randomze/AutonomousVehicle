
x = -12.77;
y = 60;
window = 20;

% generate and plot car
figure(1);
clf;
car = car_repr(x, y, pi/4, pi/6, L, Lr, Lf, d, r, 0.1); 
plot_car(car, {'-b', 'LineWidth', 1.2});

% generate and plot blocks
blcks = load_blocks(road_edges, meters_per_pixel); 
ad_blocks = admissible_blocks(blcks, [x-window/2, x+window/2], [y-window/2, y+window/2]); 
plot_blocks(ad_blocks, meters_per_pixel); 
axis equal;

% actually only need to check for collisions between blocks in neighborhood
% of car and car (the following is already too much):
tic; collide = car_colliding_blocks(car, blcks, meters_per_pixel, [x, y], Lr+L+Lf+d)
toc % on the order of ms

% check collision between blocks in display and car
tic; collide = car_colliding_blocks(car, blcks, meters_per_pixel, [x, y], window)
toc % 2-10 times slower
% equivalent to
% collide = car_colliding_blocks(car, ad_blocks, meters_per_pixel)

% SLOW alternative is checking collision for all blocks
% tic; collide = car_colliding_blocks(car, blcks, meters_per_pixel);
% toc % >1000 times slower


