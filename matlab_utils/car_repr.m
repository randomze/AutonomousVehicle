function car = car_repr(x, y, theta, phi, L, Lr, Lf, d, r, wheel_width)
%CAR_REPR Returns the car's decomposition in rectangles in fixed frame
%   Returns an array of four rectangles (body, wheels front, left, right)
%   These rectangles are represented by their edges (anti-clockwise)
    car_body = zeros(4, 4, 2);
    car = zeros(4, 4, 2);
    
    % First create the layout of the car in body frame
    
    % Body
    car_body(1, :, :) = get_rect_edges([-Lr, -d], Lr+L+Lf, 2*d);

    % Front wheels
    wheel_f_in_its_frame = get_rect_edges([-r, -wheel_width/2], 2*r, wheel_width);
    
    front_wheel_rotation_matrix = [[ cos(phi), -sin(phi)];
                                   [ sin(phi),  cos(phi)]];
    front_wheel_translation = [L, 0]';
    wheel_f_body_frame = zeros(4, 2);
    for i = 1:length(wheel_f_in_its_frame)
        wheel_f_body_frame(i, :) = front_wheel_rotation_matrix*wheel_f_in_its_frame(i, :)' + front_wheel_translation;
    end
    car_body(2, :, :) = wheel_f_body_frame(:, :);
    
    % Rear wheels
    car_body(3, :, :) = get_rect_edges([-r, d], 2*r, wheel_width);
    car_body(4, :, :) = get_rect_edges([-r, -d - wheel_width], 2*r, wheel_width);

    % Transform all to fixed frame coordinates
    rotation_body_to_fixed = [[ cos(theta), -sin(theta)];
                              [ sin(theta),  cos(theta)]];
    translation_body_to_fixed = [x, y]';
    for i = 1:4
        for j = 1:4
            car(i, j, :) = rotation_body_to_fixed*[car_body(i, j, 1), car_body(i, j, 2)]' + translation_body_to_fixed;
        end
    end
end

