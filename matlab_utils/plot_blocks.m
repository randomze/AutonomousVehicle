function plot_blocks(blocks, size, plot_args)
%PLOT_BLOCKS Plots blocks as solid squared
%   Detailed explanation goes here
    if nargin < 3
        plot_args = {'EdgeColor', 'k', 'LineWidth',2};
    end
    for i = 1:length(blocks)
        rectangle('Position',[blocks(i, 1),blocks(i, 2),size, size],'Parent',gca,plot_args{:});
    end
    
end

