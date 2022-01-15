function plot_car(car, plot_args)
%PLOT_CAR Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 2
        plot_args = {'b'};
    end
    car_size = size(car);
    hold on;
    for i = 1:car_size(1)
        plot([car(i, :, 1), car(i, 1, 1)], [car(i, :, 2), car(i, 1, 2)], plot_args{:});
    end
end

