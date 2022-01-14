function ad_blocks = admissible_blocks(blocks, xlims, ylims)
%ADMISSIBLE_BLOCKS Gets the blocks within the rectangle delimited by
% xlims and ylims. Does NOT take advantage of blocks being sorted.
    ad_blocks = blocks(blocks(:,1) >= min(xlims) & blocks(:,1) <= max(xlims) & blocks(:,2) >= min(ylims) & blocks(:,2) <= max(ylims),:);
end

