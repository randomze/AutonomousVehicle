function collision = car_colliding_blocks(car, blocks, block_size, centerxy, window)
%CAR_COLLIDING_BLOCKS Checks if car is colliding with any block in blocks
%   that are inside window with centerxy as center
    if nargin < 5 % if no window, calculate collision with all blocks
        ad_blocks = blocks;
    else
        xlims = [centerxy(1)-window/2, centerxy(1)+window/2];
        ylims = [centerxy(2)-window/2, centerxy(2)+window/2];
        ad_blocks = admissible_blocks(blocks, xlims, ylims);
    end

    car_size = size(car);
    hold on;
    for i = 1:car_size(1)
        car_part = reshape(car(i, :, :), [], 2);
        for j = 1:length(ad_blocks)
            block = ad_blocks(j, :);
            block_as_edges = get_rect_edges(block, block_size, block_size);
            if is_colliding(car_part, block_as_edges)
                collision = true;
                return
            end
        end
    end
    collision = false;
end

