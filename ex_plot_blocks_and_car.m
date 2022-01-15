
x = -15;
y = 60;
window = 50;

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

