function edges = get_rect_edges(pos_bottom_left, sidex, sidey)
%GET_EDGES Returns de x and y coordinates of the adjacent vertexes of the block
    if nargin < 3
        sidey = sidex;
    end
    edges = zeros(4, 2);
    x = pos_bottom_left(1);
    y = pos_bottom_left(2);
    edges(1,:) = [x, y];
    edges(2,:) = [x + sidex, y];
    edges(3,:) = [x + sidex, y + sidey];
    edges(4,:) = [x, y + sidey];
end

