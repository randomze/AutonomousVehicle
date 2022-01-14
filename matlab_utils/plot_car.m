function plot_car(car)
%PLOT_CAR Summary of this function goes here
%   Detailed explanation goes here
    car_size = size(car);
    color = 'b';
    hold on;
    for i = 1:car_size(1)
        plot(car(i, :, 1), car(i, :, 2), color);
        plot([car(i, 1, 1), car(i, end, 1)], [car(i, 1, 2), car(i, end, 2)], color);
    end
end

% function plot_car(car)
% %PLOT_CAR Summary of this function goes here
% %   Detailed explanation goes here
%     car_size = size(car);
%     hold on;
%     for i = 1:car_size(1)
%         for j = 1:car_size(2)-1
%             plot([car(i, j, 1), car(i, j+1, 1)], [car(i, j, 2), car(i, j+1, 2)]);
%         end
%         plot([car(i, 1, 1), car(i, end, 1)], [car(i, 1, 2), car(i, end, 2)]);
%     end
% end
% 
