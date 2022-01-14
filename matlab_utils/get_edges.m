function edges = get_edges(block,scale)
%GET_EDGES Returns de x and y coordinates of the adjacent vertexes of the block
    edges = zeros(5, 2);
    [x, y] = block(:);
    edges(1,:) = [x, y];
    edges(2,:) = [x + scale, y];
    edges(3,:) = [x + scale, y + scale];
    edges(4,:) = [x, y + scale];
end

