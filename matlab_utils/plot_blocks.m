function plot_blocks(blocks, size)
%PLOT_BLOCKS Plots blocks as solid squared
%   Detailed explanation goes here
    
    for i = 1:length(blocks)
        rectangle('Position',[blocks(i, 1),blocks(i, 2),size, size],'EdgeColor','k','LineWidth',2,'Parent',gca);
    end
    
end

