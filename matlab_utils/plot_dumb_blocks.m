function plot_dumb_blocks(road, road_img, xl, yl)
%PLOT_DUMB_BLOCKS Summary of this function goes here
%   Detailed explanation goes here
    s = road.meters_per_pixel; S = size(road_img);
    xm = xl(1); xM = xl(2);
    ym = yl(1); yM = yl(2);

    edg_botleft = [fix(xm/s)*s+s, fix(ym/s)*s+s];
    edg_topright = [fix(xM/s)*s, fix(yM/s)*s];

    bottom_left_relative_x = (edg_botleft(1)-xm)/(xM-xm);
    bottom_left_relative_y = (edg_botleft(2)-ym)/(yM-ym);
    width = (edg_topright(1)-edg_botleft(1))/(xM-xm);
    height = (edg_topright(2)-edg_botleft(2))/(yM-ym);

    pixels_min_x = S(1) - (round(edg_topright(2)/s) + S(1)/2);
    pixels_max_x = S(1) - (round(edg_botleft(2)/s) + S(1)/2);
    pixels_min_y = (round(edg_botleft(1)/s) + S(2)/2);
    pixels_max_y = (round(edg_topright(1)/s) + S(2)/2);

    ax = findobj(gcf,'Type','axes');
    cur_pos = get(ax, 'Position');
    newpos(1) = cur_pos(1)+bottom_left_relative_x;
    newpos(2) = cur_pos(2)+bottom_left_relative_y;
    set(ax,'Position', [newpos(1) newpos(2) width*cur_pos(3) height*cur_pos(3)]);
    
    pixels_min_x = max(pixels_min_x, 1);
    pixels_max_x = min(pixels_max_x, S(1));
    pixels_min_y = max(pixels_min_y, 1);
    pixels_max_y = min(pixels_max_y, S(2));

    to_plot = road_img(pixels_min_x:pixels_max_x, pixels_min_y:pixels_max_y);
    RI = imref2d(size(to_plot));
    RI.XWorldLimits = xl;
    RI.YWorldLimits = yl;
    imshow(to_plot, RI);
    
end

