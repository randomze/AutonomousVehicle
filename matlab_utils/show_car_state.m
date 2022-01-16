function show_car_state(car, car_s, road, road_img, plot_block_args, plot_car_args)
%PLOT_CAR_STATE Makes a simple video with the car's state and road

    if nargin < 6
        plot_block_args = {'EdgeColor', 'k', 'LineWidth',2};
    end
    if nargin < 5
        plot_car_args = {'-b', 'LineWidth', 1.2};
    end

    video = VideoWriter('car_state.avi');
    open(video)
    window = car.Length*10;
    sz = length(car_s.x.Data);

    f = waitbar(0, sprintf('Building video. 0/%d', sz));

    for i = 1:sz
        clf;
        x = car_s.x.Data(i);
        y = car_s.y.Data(i);
        theta = car_s.theta.Data(i);
        phi = car_s.phi.Data(i);

        xl = [x-window/2, x+window/2];
        yl = [y-window/2, y+window/2];
        xlim(xl);
        ylim(yl);
        
        plot_dumb_blocks(road, ~road_img, xl, yl);

        car_r = car_repr(x, y, theta, phi, car.L, car.Lr, car.Lf, car.d, car.r, 0.1); 
        plot_car(car_r, plot_car_args);

%         ad_blocks = admissible_blocks(blocks, xl, yl); 
%         plot_blocks(ad_blocks, road.meters_per_pixel, plot_block_args); 

        axis equal;
        frame = getframe(gcf);
        writeVideo(video, frame);
        waitbar(i/sz, f, sprintf('Building video. %d/%d', i, sz));
    end
end

