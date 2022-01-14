function plot_blocks(blocks, size, EdgeColor)
%PLOT_BLOCKS Plots blocks as solid squared
%   Detailed explanation goes here
    if nargin < 3
        EdgeColor = 'k';
    end
    for i = 1:length(blocks)
        rectangle('Position',[blocks(i, 1),blocks(i, 2),size, size],'EdgeColor',EdgeColor,'LineWidth',2,'Parent',gca);
    end
    
end

