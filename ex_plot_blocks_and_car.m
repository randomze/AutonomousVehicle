
x = 370;
y = -315;
window = 100;

% one-liner to generate and plot car
figure(1); 
clf;
car = car_repr(x, y, pi/4, pi/6, L, Lr, Lf, d, r, 0.1); 
plot_car(car); 
axis equal;

% generate and plot blocks
blcks = load_blocks(road_edges, meters_per_pixel); 
ad_blocks = admissible_blocks(blcks, [x-window, x+window], [y-window, y+window]); 
plot_blocks(ad_blocks, meters_per_pixel); 
axis equal;

