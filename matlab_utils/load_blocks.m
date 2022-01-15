function blocks = load_blocks(edge_binary_img, scale)
%LOAD_EDGES Creates blocks from image file - returns a sorted array
%   Blocks are composed of their x, y coordinate - their sidelength is the
%   scale
    % As many blocks as 1-valued pixels in the edge image
    n_blocks = sum(edge_binary_img(:));

    % Initialize blocks
    blocks = zeros(n_blocks, 2);

    % Find the 1-valued pixels
    [y, x] = find(edge_binary_img);
    
    % Get information to center image
    img_sz = size(edge_binary_img);
    center_x = img_sz(2)/2;
    center_y = img_sz(1)/2;

    % For each 1-valued pixel, create the corresponding block
    for i = 1:n_blocks
        % Create the block with the xy coordinates, inverting y coordinate
        % to keep usual (x - left, y - up) coordinate system
        blocks(i, :) = scale*[(x(i)-center_x), -(y(i)-center_y)];
    end

end
